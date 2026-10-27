use std::{
    ffi::CStr,
    fs::File,
    io::Write,
    os::unix::io::{AsFd, AsRawFd, BorrowedFd, RawFd},
};

/// A file whose fd cannot be written by other processes
///
/// This mechanism is useful for giving clients access to large amounts of
/// information such as keymaps without them being able to write to the handle.
///
/// On Linux, Android, and FreeBSD, this uses a sealed memfd. On other platforms
/// it creates a POSIX shared memory object with `shm_open`, opens a read-only
/// copy, and unlinks it.
#[derive(Debug)]
pub struct SealedFile {
    file: File,
    size: usize,
}

impl SealedFile {
    /// Create a `[SealedFile]` with the given nul-terminated C string.
    pub fn with_content(name: &CStr, contents: &CStr) -> Result<Self, std::io::Error> {
        Self::with_data(name, contents.to_bytes_with_nul())
    }

    /// Create a `[SealedFile]` with the given binary data.
    #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "android"))]
    pub fn with_data(name: &CStr, data: &[u8]) -> Result<Self, std::io::Error> {
        use rustix::fs::{MemfdFlags, SealFlags};
        use std::io::Seek;

        let fd = rustix::fs::memfd_create(name, MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING)?;

        let mut file: File = fd.into();
        file.write_all(data)?;
        file.flush()?;

        file.seek(std::io::SeekFrom::Start(0))?;

        rustix::fs::fcntl_add_seals(
            &file,
            SealFlags::SEAL | SealFlags::SHRINK | SealFlags::GROW | SealFlags::WRITE,
        )?;

        Ok(Self {
            file,
            size: data.len(),
        })
    }

    /// Create a `[SealedFile]` with the given binary data.
    #[cfg(not(any(target_os = "linux", target_os = "freebsd", target_os = "android")))]
    pub fn with_data(name: &CStr, data: &[u8]) -> Result<Self, std::io::Error> {
        use rand::{Rng, distr::Alphanumeric};
        use rustix::{
            io::Errno,
            shm::{self, Mode},
        };

        let mut rng = rand::rng();

        // `memfd_create` isn't available. Instead, try `shm_open` with a randomized name, and
        // loop a couple times if it exists.
        let mut n = 0;
        let (shm_name, mut file) = loop {
            let mut shm_name = name.to_bytes().to_owned();
            shm_name.push(b'-');
            shm_name.extend((0..7).map(|_| rng.sample(Alphanumeric)));
            let fd = shm::open(
                shm_name.as_slice(),
                shm::OFlags::RDWR | shm::OFlags::CREATE | shm::OFlags::EXCL,
                Mode::RWXU,
            );
            if !matches!(fd, Err(Errno::EXIST)) || n > 3 {
                break (shm_name, File::from(fd?));
            }
            n += 1;
        };

        // Sealing isn't available, so re-open read-only.
        let fd_rdonly = shm::open(shm_name.as_slice(), shm::OFlags::RDONLY, Mode::empty())?;
        let file_rdonly = File::from(fd_rdonly);

        // Unlink so another process can't open shm file.
        let _ = shm::unlink(shm_name.as_slice());

        file.write_all(data)?;
        file.flush()?;

        Ok(Self {
            file: file_rdonly,
            size: data.len(),
        })
    }

    /// Open an independent read-only copy of the sealed data: a new open
    /// file description with its own offset at 0.
    ///
    /// The sealed file is one file with one offset, and `SCM_RIGHTS` hands
    /// every receiver that same description, offset included: the first
    /// `read()`-based `wl_keyboard` client to read moves it to the end and
    /// every later client reads zero bytes (toolkits `mmap`, which ignores
    /// offsets, and never notice). Reopening through `/proc/self/fd` gives
    /// each keymap send its own description, so readers never disturb each
    /// other. Seals live on the file rather than the description, so the
    /// copy stays sealed; it is closed once the send returns, and the
    /// receiver's own copy outlives it.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn open_independent(&self) -> std::io::Result<File> {
        let path = format!("/proc/self/fd/{}", self.file.as_raw_fd());
        let owned = rustix::fs::openat(
            rustix::fs::CWD,
            path.as_str(),
            rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?;
        Ok(File::from(owned))
    }

    /// Size of the data contained in the sealed file.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl AsRawFd for SealedFile {
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

impl AsFd for SealedFile {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
mod tests {
    use super::*;
    use std::io::Read;

    /// Two independent copies must not share an offset: reading one to the
    /// end leaves the other (and the sealed file itself) at 0. This is the
    /// property `wl_keyboard.keymap` sends rely on.
    #[test]
    fn independent_copies_do_not_share_an_offset() {
        let sealed = SealedFile::with_data(c"test-keymap", b"0123456789").expect("a sealed file");
        let mut first = sealed.open_independent().expect("a first copy");
        let mut second = sealed.open_independent().expect("a second copy");

        let mut first_bytes = Vec::new();
        first.read_to_end(&mut first_bytes).expect("a full read");
        assert_eq!(first_bytes, b"0123456789");

        let mut second_bytes = Vec::new();
        second
            .read_to_end(&mut second_bytes)
            .expect("a full read after the first copy was drained");
        assert_eq!(second_bytes, b"0123456789");

        // The sealed file's own offset is untouched: no seek, still a full
        // read. A clone shares its description, which is exactly what makes
        // this an assertion rather than a tautology.
        let mut sealed_bytes = Vec::new();
        sealed
            .file
            .try_clone()
            .expect("a clone shares the offset")
            .read_to_end(&mut sealed_bytes)
            .expect("the sealed file itself is untouched");
        assert_eq!(sealed_bytes, b"0123456789");
    }
}
