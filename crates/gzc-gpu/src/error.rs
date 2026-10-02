//! The error type of the [`crate::Compressor`] API.
//!
//! Inside the crate, functions return `anyhow::Result`. A failure that the caller can act on is
//! built with [`tagged`], which puts a [`Kind`] into the error's chain. [`Error::from_anyhow`]
//! reads it back at the API boundary.
use std::fmt;

/// Why a [`crate::Compressor`] call failed.
///
/// The variants tell apart what a caller can do about it:
///
/// - [`Error::NoAdapter`], [`Error::DeviceLost`]: compress on the CPU instead, or retry with a
///   new compressor.
/// - [`Error::OutOfMemory`]: use a smaller VRAM budget or batch.
/// - [`Error::Unsupported`], [`Error::InvalidInput`]: fix the options or the input.
///
/// Each variant holds a message that names what failed. `Display` prints it.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// No usable GPU adapter was found.
    NoAdapter(String),
    /// The match parameters are valid, but the GPU kernels do not implement them.
    Unsupported(String),
    /// A GPU allocation failed. The device stays usable.
    OutOfMemory(String),
    /// The GPU device was lost. Every later call on the same compressor fails too.
    DeviceLost(String),
    /// An argument or option is out of range: a block that is empty or longer than 64 KiB,
    /// invalid match parameters, zero batches in flight, a batch that does not fit the device or
    /// the budget.
    InvalidInput(String),
    /// Any other failure, with its cause. An error that a caller's own closure returned to
    /// [`crate::Compressor::stream`] comes back unchanged, whatever its variant.
    Other(Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAdapter(m)
            | Error::Unsupported(m)
            | Error::OutOfMemory(m)
            | Error::DeviceLost(m)
            | Error::InvalidInput(m) => f.write_str(m),
            Error::Other(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Other(e) => e.source(),
            _ => None,
        }
    }
}

/// The class of a tagged internal error; one per [`Error`] variant but `Other`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    NoAdapter,
    Unsupported,
    OutOfMemory,
    DeviceLost,
    InvalidInput,
}

/// An error message with its class.
#[derive(Debug)]
struct Tagged {
    kind: Kind,
    msg: String,
}

impl fmt::Display for Tagged {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl std::error::Error for Tagged {}

/// An `anyhow` error that reads as `msg` and converts to the `kind` variant of [`Error`].
pub(crate) fn tagged(kind: Kind, msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Tagged { kind, msg: msg.into() })
}

/// `tagged(Kind::InvalidInput, ..)`.
pub(crate) fn invalid_input(msg: impl Into<String>) -> anyhow::Error {
    tagged(Kind::InvalidInput, msg)
}

impl Error {
    /// The typed form of an internal error.
    ///
    /// An [`Error`] that travelled through `anyhow` (a caller's closure returned it) comes back
    /// as it was. Otherwise the first tag in the chain picks the variant, and the message is the
    /// whole chain. An untagged error becomes [`Error::Other`].
    pub(crate) fn from_anyhow(e: anyhow::Error) -> Self {
        let e = match e.downcast::<Error>() {
            Ok(own) => return own,
            Err(e) => e,
        };
        let kind = e.chain().find_map(|c| c.downcast_ref::<Tagged>()).map(|t| t.kind);
        let msg = || format!("{e:#}");
        match kind {
            Some(Kind::NoAdapter) => Error::NoAdapter(msg()),
            Some(Kind::Unsupported) => Error::Unsupported(msg()),
            Some(Kind::OutOfMemory) => Error::OutOfMemory(msg()),
            Some(Kind::DeviceLost) => Error::DeviceLost(msg()),
            Some(Kind::InvalidInput) => Error::InvalidInput(msg()),
            None => Error::Other(e.into_boxed_dyn_error()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;

    #[test]
    fn tags_survive_context_and_pick_the_variant() {
        let e = Err::<(), _>(tagged(Kind::OutOfMemory, "GPU allocation of 3 MiB failed")).context("opening").unwrap_err();
        match Error::from_anyhow(e) {
            Error::OutOfMemory(m) => assert_eq!(m, "opening: GPU allocation of 3 MiB failed"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(Error::from_anyhow(invalid_input("block 3 is empty")), Error::InvalidInput(_)));
        assert!(matches!(Error::from_anyhow(tagged(Kind::NoAdapter, "none")), Error::NoAdapter(_)));
        assert!(matches!(Error::from_anyhow(tagged(Kind::DeviceLost, "lost")), Error::DeviceLost(_)));
        assert!(matches!(Error::from_anyhow(tagged(Kind::Unsupported, "no")), Error::Unsupported(_)));
    }

    #[test]
    fn untagged_errors_keep_their_source() {
        let io = std::io::Error::other("disk full");
        let e = Error::from_anyhow(anyhow::Error::new(io).context("writing a frame"));
        assert!(matches!(e, Error::Other(_)));
        assert_eq!(e.to_string(), "writing a frame");
        assert_eq!(std::error::Error::source(&e).unwrap().to_string(), "disk full");
    }

    #[test]
    fn a_callers_own_error_passes_through() {
        let own = Error::InvalidInput("mine".to_string());
        let back = Error::from_anyhow(anyhow::Error::new(own).context("stream failed"));
        assert!(matches!(back, Error::InvalidInput(m) if m == "mine"));
    }
}
