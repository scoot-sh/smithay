use std::{
    collections::HashMap,
    fmt,
    os::fd::{BorrowedFd, OwnedFd},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use calloop::{LoopHandle, RegistrationToken};
use rustix::{
    event::{PollFd, PollFlags, poll},
    time::Timespec,
};
use smallvec::SmallVec;
use tracing::{debug, trace, warn};
use x11rb::{
    connection::Connection as _,
    errors::ReplyOrIdError,
    protocol::{
        xfixes::{ConnectionExt as _, SelectionEventMask},
        xproto::{
            Atom, AtomEnum, ConnectionExt as _, CreateWindowAux, EventMask, PropMode, SELECTION_NOTIFY_EVENT,
            Screen, SelectionNotifyEvent, SelectionRequestEvent, Window as X11Window, WindowClass,
        },
    },
    rust_connection::RustConnection,
    wrapper::ConnectionExt as _,
};

use crate::{
    wayland::selection::SelectionTarget,
    xwayland::xwm::{Atoms, OwnedX11Window},
};

// copied from wlroots - docs say "maximum size can vary widely depending on the implementation"
// and there is no way to query the maximum size, you just get a non-descriptive `Length` error...
pub const INCR_CHUNK_SIZE: usize = 64 * 1024;

/// The most transfers of one selection into Wayland clients that may wait on their X owner's
/// answer at once, and separately the most that may be under way (answered, streaming to their
/// reader) at once. Each holds a file descriptor and a window, and an answered one a slice of
/// data; a paste and a clipboard manager reading together is the usual worst case. A read past
/// either bound is refused -- the reader sees an empty transfer -- not queued.
pub const MAX_SELECTION_TRANSFERS: usize = 8;

/// A transfer, either way, that has not moved for this long is dropped the next time the window
/// manager looks (a new request, a new answer, a change of owner). The default for
/// [`X11Wm::set_selection_transfer_timeout`](super::X11Wm::set_selection_transfer_timeout).
pub const SELECTION_TRANSFER_TIMEOUT: Duration = Duration::from_secs(30);

/// Once a selection has changed hands (or its owner has gone), a transfer started under the
/// previous owner that is waiting on it for its next chunk, idle for at least this long, is
/// dropped. One that is still moving is left to finish: ending it would hand the reader a
/// partial selection as if it were whole.
pub const OWNER_CHANGE_GRACE: Duration = Duration::from_secs(1);

/// How often the window manager sweeps selection transfers while any are in flight (the timer
/// is not armed at all while none are).
pub const SWEEP_INTERVAL: Duration = Duration::from_secs(1);

/// A conversion of an X selection, requested for a Wayland reader, that its owner has not
/// answered yet.
#[derive(Debug)]
pub struct PendingTransfer {
    pub window: OwnedX11Window,
    /// The reader's end.
    pub fd: OwnedFd,
    pub since: Instant,
}

impl PendingTransfer {
    pub fn new(window: OwnedX11Window, fd: OwnedFd) -> Self {
        PendingTransfer {
            window,
            fd,
            since: Instant::now(),
        }
    }
}

/// Whether the reader holding the other end of the pipe `fd` writes into has closed it: the
/// pipe reports an error or hang-up.
fn reader_gone(fd: &OwnedFd) -> bool {
    let mut fds = [PollFd::new(fd, PollFlags::OUT)];
    match poll(&mut fds, Some(&Timespec::default())) {
        Ok(_) => fds[0].revents().intersects(PollFlags::ERR | PollFlags::HUP),
        Err(_) => false,
    }
}

/// The most transfers out of one selection (a Wayland selection read by X clients) one X client
/// may have in flight at once. Each holds a pipe and up to two chunks of data until the
/// requestor has taken it; a client pasting reads one or two targets at a time.
pub const MAX_OUTGOING_PER_CLIENT: usize = 4;

/// The most transfers out of one selection in flight at once, across all X clients.
pub const MAX_OUTGOING_TRANSFERS: usize = 16;

/// How much of a selection property an incoming transfer reads at a time, in 32-bit units
/// (`GetProperty` counts in those): one [`INCR_CHUNK_SIZE`]. The next slice is read only once
/// this one has been written into the reader's pipe, so a transfer never buffers more than one
/// slice, however large the owner makes the property.
pub const PROPERTY_SLICE: u32 = (INCR_CHUNK_SIZE / 4) as u32;

#[derive(Debug)]
pub struct XWmSelection {
    pub atom: Atom,

    pub conn: Arc<RustConnection>,
    pub atoms: Atoms,
    pub window: OwnedX11Window,
    pub owner: X11Window,
    /// Bumped on every ownership change the window manager hears of (see
    /// [`X11Wm::selection_generation`](super::X11Wm::selection_generation)).
    pub generation: u64,
    pub mime_types: Vec<String>,
    pub timestamp: u32,

    pub pending_transfers: Arc<Mutex<HashMap<X11Window, PendingTransfer>>>,
    pub incoming: HashMap<X11Window, IncomingTransfer>,
    pub outgoing: HashMap<X11Window, OutgoingTransfer>,
}

pub struct IncomingTransfer {
    pub token: Option<RegistrationToken>,
    pub window: OwnedX11Window,
    /// The reader's end, shared with the write source, so a sweep can see a reader that left.
    pub fd: Arc<OwnedFd>,

    pub incr: bool,
    /// Read from the property, not yet written to the reader: at most one slice.
    pub source_data: Vec<u8>,
    /// Where the next slice of the current property starts, in 32-bit units.
    pub offset: u32,
    /// The current property has more past `offset`.
    pub more: bool,
    /// INCR: the delete asking for the next chunk has been sent, and its `PropertyNotify`
    /// not yet seen.
    pub delete_sent: bool,
    /// INCR: the delete has taken effect, so the owner's next write is the next chunk. Any
    /// new value seen while this is unset -- an owner appending without waiting, or a
    /// notify queued before the delete -- is ignored, never read.
    pub awaiting_chunk: bool,
    /// When the transfer last moved: a slice read, bytes written, a chunk arriving.
    pub last_activity: Instant,
    /// The selection's ownership count when the transfer was answered: a transfer whose owner
    /// has since changed or gone can stall for good.
    pub generation: u64,
}

impl fmt::Debug for IncomingTransfer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IncomingTransfer")
            .field("token", &self.token)
            .field("window", &self.window)
            .field("incr", &self.incr)
            .field("buffered", &self.source_data.len())
            .field("offset", &self.offset)
            .field("more", &self.more)
            .field("delete_sent", &self.delete_sent)
            .field("awaiting_chunk", &self.awaiting_chunk)
            .finish()
    }
}

impl IncomingTransfer {
    /// A transfer into `fd` through `window`, not started.
    pub fn new(token: RegistrationToken, window: OwnedX11Window, fd: Arc<OwnedFd>, generation: u64) -> Self {
        IncomingTransfer {
            token: Some(token),
            window,
            fd,
            incr: false,
            source_data: Vec::new(),
            offset: 0,
            more: false,
            delete_sent: false,
            awaiting_chunk: false,
            last_activity: Instant::now(),
            generation,
        }
    }

    /// Reads the next slice of the property into the buffer, returning the property's type.
    pub fn read_slice(&mut self, conn: &RustConnection, atoms: &Atoms) -> Result<Atom, ReplyOrIdError> {
        let reply = conn
            .get_property(
                false,
                *self.window,
                atoms._WL_SELECTION,
                AtomEnum::ANY,
                self.offset,
                PROPERTY_SLICE,
            )?
            .reply()?;
        // A full slice is a whole number of 32-bit units; only the last may not be, and
        // nothing is read after it.
        let units = u32::try_from(reply.value.len() / 4).unwrap_or(u32::MAX);
        self.offset = self.offset.saturating_add(units);
        self.more = reply.bytes_after > 0 && self.offset < u32::MAX;
        self.source_data.extend_from_slice(&reply.value);
        self.last_activity = Instant::now();
        Ok(reply.type_)
    }

    pub fn write_selection(&mut self, fd: BorrowedFd<'_>) -> std::io::Result<bool> {
        if self.source_data.is_empty() {
            return Ok(true);
        }

        let len = rustix::io::write(fd, &self.source_data)?;
        if len > 0 {
            self.source_data.drain(..len);
            self.last_activity = Instant::now();
        }

        Ok(self.source_data.is_empty())
    }

    pub fn destroy<D>(mut self, handle: &LoopHandle<'_, D>) {
        if let Some(token) = self.token.take() {
            handle.remove(token);
        }
    }

    /// Whether the reader has closed its end: the pipe reports an error or hang-up. A transfer
    /// waiting on its owner writes nothing, so without asking it would never notice.
    pub fn reader_gone(&self) -> bool {
        reader_gone(&self.fd)
    }

    /// Whether the transfer is waiting on the owner for its next chunk.
    pub fn waiting_on_owner(&self) -> bool {
        self.delete_sent || self.awaiting_chunk
    }
}

impl Drop for IncomingTransfer {
    fn drop(&mut self) {
        if self.token.is_some() {
            tracing::warn!(
                ?self,
                "IncomingTransfer freed before being removed from EventLoop"
            );
        }
    }
}

pub struct OutgoingTransfer {
    pub conn: Arc<RustConnection>,
    pub token: Option<RegistrationToken>,

    pub incr: bool,
    pub source_data: Vec<u8>,
    pub request: SelectionRequestEvent,

    pub property_set: bool,
    pub flush_property_on_delete: bool,
    /// The final 0-byte data chunk has been sent, denoting the completion of this transfer
    pub sent_finished: bool,
    /// When the transfer last moved: bytes read from the source, or a chunk taken.
    pub last_activity: Instant,
}

impl fmt::Debug for OutgoingTransfer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutgoingTransfer")
            .field("conn", &"...")
            .field("token", &self.token)
            .field("incr", &self.incr)
            .field("source_data", &self.source_data)
            .field("request", &self.request)
            .field("property_set", &self.property_set)
            .field("flush_property_on_delete", &self.flush_property_on_delete)
            .finish()
    }
}

impl OutgoingTransfer {
    pub fn flush_data(&mut self) -> Result<usize, ReplyOrIdError> {
        let len = std::cmp::min(self.source_data.len(), INCR_CHUNK_SIZE);

        if len == 0 {
            // This flush will complete the transfer
            self.sent_finished = true;
        }

        let mut data = self.source_data.split_off(len);
        std::mem::swap(&mut data, &mut self.source_data);

        self.conn.change_property8(
            PropMode::REPLACE,
            self.request.requestor,
            self.request.property,
            self.request.target,
            &data,
        )?;
        self.conn.flush()?;

        let remaining = self.source_data.len();
        self.property_set = true;
        Ok(remaining)
    }

    pub fn destroy<D>(mut self, handle: &LoopHandle<'_, D>) {
        if let Some(token) = self.token.take() {
            handle.remove(token);
        }
    }
}

impl Drop for OutgoingTransfer {
    fn drop(&mut self) {
        if self.token.is_some() {
            tracing::warn!(
                ?self,
                "OutgoingTransfer freed before being removed from EventLoop"
            );
        }
    }
}

impl XWmSelection {
    pub fn new(
        conn: &Arc<RustConnection>,
        screen: &Screen,
        atoms: &Atoms,
        atom: Atom,
    ) -> Result<Self, ReplyOrIdError> {
        let window = conn.generate_id()?;
        conn.create_window(
            screen.root_depth,
            window,
            screen.root,
            0,
            0,
            10,
            10,
            0,
            WindowClass::INPUT_OUTPUT,
            screen.root_visual,
            &CreateWindowAux::new().event_mask(EventMask::PROPERTY_CHANGE),
        )?;

        if atom == atoms.CLIPBOARD {
            conn.set_selection_owner(window, atoms.CLIPBOARD_MANAGER, x11rb::CURRENT_TIME)?;
        }
        conn.xfixes_select_selection_input(
            window,
            atom,
            SelectionEventMask::SET_SELECTION_OWNER
                | SelectionEventMask::SELECTION_WINDOW_DESTROY
                | SelectionEventMask::SELECTION_CLIENT_CLOSE,
        )?;
        conn.flush()?;

        debug!(
            selection_window = ?window,
            ?atom,
            "Selection init",
        );

        Ok(XWmSelection {
            atom,
            conn: conn.clone(),
            atoms: *atoms,
            window: OwnedX11Window::new(window, conn),
            owner: x11rb::NONE,
            generation: 0,
            mime_types: Vec::new(),
            timestamp: x11rb::CURRENT_TIME,
            pending_transfers: Arc::new(Mutex::new(HashMap::new())),
            incoming: HashMap::new(),
            outgoing: HashMap::new(),
        })
    }

    /// Drops the transfers that will not finish, and says whether any are left:
    ///
    /// - a conversion waiting on its owner whose reader has left, or that has waited for
    ///   `timeout`;
    /// - an incoming transfer whose reader has left, that has not moved for `timeout`, or that
    ///   is waiting on its owner for a chunk, idle past [`OWNER_CHANGE_GRACE`], after the
    ///   selection has changed hands (or lost its owner) since it was answered;
    /// - an outgoing transfer that has not moved for `timeout`.
    pub fn sweep<D>(&mut self, timeout: Duration, loop_handle: &LoopHandle<'_, D>) -> bool {
        let now = Instant::now();
        let generation = self.generation;
        {
            let mut pending = self.pending_transfers.lock().unwrap();
            pending.retain(|_, transfer| {
                now.saturating_duration_since(transfer.since) < timeout && !reader_gone(&transfer.fd)
            });
        }
        let stale: SmallVec<[X11Window; 8]> = self
            .incoming
            .iter()
            .filter(|(_, transfer)| {
                let idle = now.saturating_duration_since(transfer.last_activity);
                idle >= timeout
                    || (transfer.generation != generation
                        && transfer.waiting_on_owner()
                        && idle >= OWNER_CHANGE_GRACE)
                    || transfer.reader_gone()
            })
            .map(|(window, _)| *window)
            .collect();
        for window in stale {
            if let Some(transfer) = self.incoming.remove(&window) {
                debug!(
                    ?transfer,
                    "Dropping an incoming selection transfer that will not finish"
                );
                transfer.destroy(loop_handle);
            }
        }
        let stale: SmallVec<[X11Window; 8]> = self
            .outgoing
            .iter()
            .filter(|(_, transfer)| now.saturating_duration_since(transfer.last_activity) >= timeout)
            .map(|(window, _)| *window)
            .collect();
        for window in stale {
            if let Some(transfer) = self.outgoing.remove(&window) {
                debug!(requestor = window, "Dropping an idle outgoing selection transfer");
                transfer.destroy(loop_handle);
            }
        }
        !self.incoming.is_empty()
            || !self.outgoing.is_empty()
            || !self.pending_transfers.lock().unwrap().is_empty()
    }

    pub fn window_destroyed<D>(&mut self, window: &X11Window, loop_handle: &LoopHandle<'_, D>) -> bool {
        (if let Some(transfer) = self.incoming.remove(window) {
            transfer.destroy(loop_handle);
            true
        } else {
            false
        }) || (if let Some(transfer) = self.outgoing.remove(window) {
            transfer.destroy(loop_handle);
            true
        } else {
            false
        }) || self.pending_transfers.lock().unwrap().remove(window).is_some()
    }

    pub fn has_window(&self, window: &X11Window) -> bool {
        self.window == *window
            || self.incoming.contains_key(window)
            || self.pending_transfers.lock().unwrap().contains_key(window)
    }

    pub fn type_(&self) -> Option<SelectionTarget> {
        match self.atom {
            x if x == self.atoms.CLIPBOARD => Some(SelectionTarget::Clipboard),
            x if x == self.atoms.PRIMARY => Some(SelectionTarget::Primary),
            _ => None,
        }
    }
}

pub enum OutgoingAction {
    Done,
    DoneReading,
    WaitForReadable,
    /// A chunk is buffered and the requestor has not yet taken the one before it: stop
    /// reading until it deletes the property (see `read_selection_callback`).
    WaitForDelete,
}

pub fn read_selection_callback(
    conn: &RustConnection,
    atoms: &Atoms,
    fd: BorrowedFd<'_>,
    transfer: &mut OutgoingTransfer,
) -> Result<OutgoingAction, ReplyOrIdError> {
    // Backpressure: while the requestor still holds the last chunk, a full chunk already
    // waiting is all there is any use buffering. Reading on would pull the whole Wayland
    // source into memory for a requestor that may never take it -- nothing else bounds
    // this buffer. The source is re-enabled when the requestor deletes the property.
    if transfer.incr && transfer.property_set && transfer.source_data.len() >= INCR_CHUNK_SIZE {
        // The delete that resumes reading must also send the waiting chunk.
        transfer.flush_property_on_delete = true;
        return Ok(OutgoingAction::WaitForDelete);
    }
    let mut buf = [0; INCR_CHUNK_SIZE];
    let Ok(len) = rustix::io::read(fd, &mut buf) else {
        debug!(
            requestor = transfer.request.requestor,
            "File descriptor closed, aborting transfer."
        );
        send_selection_notify_resp(conn, &transfer.request, false)?;
        return Ok(OutgoingAction::Done);
    };
    trace!(
        requestor = transfer.request.requestor,
        "Transfer became readable, read {} bytes", len
    );

    transfer.source_data.extend_from_slice(&buf[..len]);
    transfer.last_activity = Instant::now();
    if transfer.source_data.len() >= INCR_CHUNK_SIZE {
        if !transfer.incr {
            // start incr transfer
            trace!(
                requestor = transfer.request.requestor,
                "Transfer became incremental",
            );
            conn.change_property32(
                PropMode::REPLACE,
                transfer.request.requestor,
                transfer.request.property,
                atoms.INCR,
                &[INCR_CHUNK_SIZE as u32],
            )?;
            conn.flush()?;
            transfer.incr = true;
            transfer.property_set = true;
            transfer.flush_property_on_delete = true;
            send_selection_notify_resp(conn, &transfer.request, true)?;
        } else if transfer.property_set {
            // got more bytes, waiting for property delete
            transfer.flush_property_on_delete = true;
        } else {
            // got more bytes, property deleted
            let len = transfer.flush_data()?;
            trace!(
                requestor = transfer.request.requestor,
                "Send data chunk: {} bytes", len
            );
        }
    }

    if len == 0 {
        if transfer.incr {
            debug!("Incr transfer completed");
            if !transfer.property_set {
                let len = transfer.flush_data()?;
                trace!(
                    requestor = transfer.request.requestor,
                    "Send data chunk: {} bytes", len
                );
            }
            transfer.flush_property_on_delete = true;
            Ok(OutgoingAction::DoneReading)
        } else {
            let len = transfer.flush_data()?;
            debug!("Non-Incr transfer completed with {} bytes", len);
            send_selection_notify_resp(conn, &transfer.request, true)?;
            Ok(OutgoingAction::Done)
        }
    } else {
        Ok(OutgoingAction::WaitForReadable)
    } // nothing to be done, buffered the bytes
}

pub enum IncomingAction {
    Done,
    WaitForProperty,
    WaitForWritable,
}

pub fn write_selection_callback(
    fd: BorrowedFd<'_>,
    conn: &RustConnection,
    atoms: &Atoms,
    transfer: &mut IncomingTransfer,
) -> Result<IncomingAction, ReplyOrIdError> {
    match transfer.write_selection(fd) {
        Ok(true) => {
            if transfer.more {
                // The next slice of the same property, now that this one is out.
                transfer.read_slice(conn, atoms)?;
                Ok(IncomingAction::WaitForWritable)
            } else if transfer.incr {
                // This delete asks the owner for the next chunk (reads never delete), so it
                // has to reach the server now, not whenever something else flushes.
                request_next_chunk(conn, atoms, transfer)?;
                Ok(IncomingAction::WaitForProperty)
            } else {
                debug!(?transfer, "Non-Incr Transfer complete!");
                Ok(IncomingAction::Done)
            }
        }
        Ok(false) => Ok(IncomingAction::WaitForWritable),
        Err(err) => {
            // The reader went away: nothing more of the selection has anywhere to go. The owner
            // is left waiting for a delete that will not come, which ICCCM owners time out.
            warn!(?err, "Transfer errored");
            Ok(IncomingAction::Done)
        }
    }
}

/// Deletes the transfer's property, which asks an INCR owner for its next chunk, and notes that
/// the chunk is only due once that delete is seen to take effect.
pub fn request_next_chunk(
    conn: &RustConnection,
    atoms: &Atoms,
    transfer: &mut IncomingTransfer,
) -> Result<(), ReplyOrIdError> {
    conn.delete_property(*transfer.window, atoms._WL_SELECTION)?;
    conn.flush()?;
    transfer.offset = 0;
    transfer.more = false;
    transfer.delete_sent = true;
    transfer.awaiting_chunk = false;
    Ok(())
}

pub fn send_selection_notify_resp(
    conn: &RustConnection,
    req: &SelectionRequestEvent,
    success: bool,
) -> Result<(), ReplyOrIdError> {
    conn.send_event(
        false,
        req.requestor,
        EventMask::NO_EVENT,
        SelectionNotifyEvent {
            response_type: SELECTION_NOTIFY_EVENT,
            sequence: 0,
            time: req.time,
            requestor: req.requestor,
            selection: req.selection,
            target: req.target,
            property: if success {
                req.property
            } else {
                AtomEnum::NONE.into()
            },
        },
    )?;
    conn.flush()?;
    Ok(())
}
