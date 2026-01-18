use candle_core::error::Result;
use candle_core::utils::metal_is_available;
use candle_core::{Device, utils::cuda_is_available};

pub fn load() -> Result<Device> {
    if metal_is_available() {
        Device::new_metal(0)
    } else if cuda_is_available() {
        Device::new_cuda(0)
    } else {
        Ok(Device::Cpu)
    }
}
