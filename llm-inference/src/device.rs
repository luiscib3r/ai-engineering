// src/device.rs
use candle_core::{Device, utils::{cuda_is_available, metal_is_available}};
use candle_core::error::Result;

/// Detectar el mejor device disponible para acelerar inferencia
pub fn load() -> Result<Device> {
    if metal_is_available() {
        // Mac con Apple Silicon (M1/M2/M3/M4)
        println!("🚀 Usando Metal (GPU de Apple)");
        Device::new_metal(0)
    } else if cuda_is_available() {
        // NVIDIA GPU con CUDA
        println!("🚀 Usando CUDA (GPU NVIDIA)");
        Device::new_cuda(0)
    } else {
        // Fallback: CPU
        println!("⚠️  Usando CPU (considera compilar con Metal o CUDA para 10-50x speedup)");
        Ok(Device::Cpu)
    }
}
