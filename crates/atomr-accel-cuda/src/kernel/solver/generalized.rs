//! Generalized symmetric / Hermitian eigenvalue problems.
//!
//! Solves `A x = λ B x` (itype=1), `A B x = λ x` (itype=2), or
//! `B A x = λ x` (itype=3) for an `n × n` symmetric `A` and SPD
//! `B`. f32/f64 dispatch through `cusolverDn[SD]sygvd`; the
//! Hermitian-complex `Hegvd` request is type-aliased to the same
//! launch path so adding c32/c64 in a future phase only needs
//! `SolverScalar` impls — no new request type.

use std::sync::Arc;

use cudarc::cusolver::sys as cs;
use cudarc::driver::DevicePtrMut;
use tokio::sync::oneshot;

use crate::dtype::SolverSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::envelope;
use crate::sys::cusolver::{status_to_result, SolverScalar, LIB};

use super::workspace::{check_info, ensure_workspace_bytes, lwork_bytes};
use super::{SolverCells, SolverDispatch, Uplo};

/// Eigenproblem type as defined by cuSOLVER's `cusolverEigType_t`:
/// `1: A x = λ B x`, `2: A B x = λ x`, `3: B A x = λ x`.
#[derive(Debug, Clone, Copy)]
pub enum EigType {
    Type1,
    Type2,
    Type3,
}

impl EigType {
    fn as_cusolver(self) -> cs::cusolverEigType_t {
        match self {
            EigType::Type1 => cs::cusolverEigType_t::CUSOLVER_EIG_TYPE_1,
            EigType::Type2 => cs::cusolverEigType_t::CUSOLVER_EIG_TYPE_2,
            EigType::Type3 => cs::cusolverEigType_t::CUSOLVER_EIG_TYPE_3,
        }
    }
}

pub struct SygvdRequest<T: SolverSupported> {
    pub a: GpuRef<T>,
    pub b: GpuRef<T>,
    pub n: i32,
    pub itype: EigType,
    pub uplo: Uplo,
    pub w: GpuRef<T>,
    pub compute_vectors: bool,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

/// `Hegvd` (Hermitian) is the complex sibling of `Sygvd`. Phase 1
/// supports only real dtypes, so this request currently shares the
/// same launch path as `SygvdRequest`. Promoting to a distinct
/// surface lets callers express intent today and lets us add c32/c64
/// without a SemVer break later.
pub struct HegvdRequest<T: SolverSupported> {
    pub a: GpuRef<T>,
    pub b: GpuRef<T>,
    pub n: i32,
    pub itype: EigType,
    pub uplo: Uplo,
    pub w: GpuRef<T>,
    pub compute_vectors: bool,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T> SolverDispatch for SygvdRequest<T>
where
    T: SolverSupported + SolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let SygvdRequest {
            a,
            b,
            n,
            itype,
            uplo,
            w,
            compute_vectors,
            reply,
        } = *self;
        run_sygvd::<T>(cells, a, b, n, itype, uplo, w, compute_vectors, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

impl<T> SolverDispatch for HegvdRequest<T>
where
    T: SolverSupported + SolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let HegvdRequest {
            a,
            b,
            n,
            itype,
            uplo,
            w,
            compute_vectors,
            reply,
        } = *self;
        run_sygvd::<T>(cells, a, b, n, itype, uplo, w, compute_vectors, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

fn run_sygvd<T: SolverScalar>(
    cells: SolverCells<'_>,
    a: GpuRef<T>,
    b: GpuRef<T>,
    n: i32,
    itype: EigType,
    uplo: Uplo,
    w: GpuRef<T>,
    compute_vectors: bool,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        handle,
        stream,
        completion,
        workspace,
        info,
        ..
    } = cells;

    let a_slice = match a.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let b_slice = match b.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let w_slice = match w.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut a_owned = match Arc::try_unwrap(a_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Sygvd a has multiple live references".into(),
            )));
            return;
        }
    };
    let mut b_owned = match Arc::try_unwrap(b_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Sygvd b has multiple live references".into(),
            )));
            return;
        }
    };
    let mut w_owned = match Arc::try_unwrap(w_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Sygvd w has multiple live references".into(),
            )));
            return;
        }
    };

    let fill = uplo.as_cusolver_fill();
    let jobz = if compute_vectors {
        cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR
    } else {
        cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_NOVECTOR
    };
    let itype_cs = itype.as_cusolver();

    let mut lwork = 0i32;
    {
        let h = handle.lock();
        let (a_ptr, _ga) = a_owned.device_ptr_mut(stream);
        let (b_ptr, _gb) = b_owned.device_ptr_mut(stream);
        let (w_ptr, _gw) = w_owned.device_ptr_mut(stream);
        let status = unsafe {
            T::sygvd_buffer_size(
                h.0.cu(),
                itype_cs,
                jobz,
                fill,
                n,
                a_ptr as *const T,
                n,
                b_ptr as *const T,
                n,
                w_ptr as *const T,
                &mut lwork as *mut _,
            )
        };
        drop((_ga, _gb, _gw));
        if let Err(e) = status_to_result(status, "sygvd_bufferSize") {
            let _ = reply.send(Err(e));
            return;
        }
    }
    if let Err(e) = ensure_workspace_bytes(workspace, stream, lwork_bytes::<T>(lwork)) {
        let _ = reply.send(Err(e));
        return;
    }

    a.record_write(stream);
    b.record_write(stream);
    w.record_write(stream);

    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle.lock();
        let mut ws = workspace.lock();
        let mut info_lock = info.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let (b_ptr, _g2) = b_owned.device_ptr_mut(&stream_for_check);
        let (w_ptr, _g3) = w_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g4) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g5) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            T::sygvd(
                h.0.cu(),
                itype_cs,
                jobz,
                fill,
                n,
                a_ptr as *mut T,
                n,
                b_ptr as *mut T,
                n,
                w_ptr as *mut T,
                ws_ptr as *mut T,
                lwork,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3, _g4, _g5));
        status_to_result(status, "sygvd")?;
        check_info(info, &stream_for_check, "sygvd")?;
        Ok((a_owned, b_owned, w_owned))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sygvd_request_round_trip() {
        fn assert_dispatch<R: SolverDispatch>() {}
        assert_dispatch::<SygvdRequest<f32>>();
        assert_dispatch::<SygvdRequest<f64>>();
        assert_dispatch::<HegvdRequest<f32>>();
        assert_dispatch::<HegvdRequest<f64>>();
    }
}
