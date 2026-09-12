//! # Inter-Process Communication (IPC)
//!
//! Handles communication between the Rust core and the Python Calibration
//! Pipeline / Inference API server using fast named pipes / unix sockets.

use crate::{PhantomError, PhantomResult};
use interprocess::local_socket::{tokio::LocalSocketStream, NameTypeSupport};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// IPC Message payload format.
#[derive(Debug, Serialize, Deserialize)]
pub enum IpcMessage {
    /// Request to load a model and profile
    LoadModel { model_path: String, profile_path: String },
    /// Status response
    Status { ready: bool, memory_used_mb: f32 },
    /// Request generation
    Generate { prompt: String, max_tokens: u32, temp: f32 },
    /// Stream generation delta
    Token { text: String, is_final: bool },
    /// Error response
    Error { message: String },
}

pub struct IpcClient {
    stream: LocalSocketStream,
}

impl IpcClient {
    /// Connect to the IPC server.
    pub async fn connect(name: &str) -> PhantomResult<Self> {
        let name = {
            use interprocess::local_socket::ToNsName;
            name.to_ns_name::<NameTypeSupport>().map_err(|e| {
                PhantomError::IpcError(format!("Invalid IPC name: {}", e))
            })?
        };

        let stream = LocalSocketStream::connect(name).await.map_err(|e| {
            PhantomError::IpcError(format!("Failed to connect to IPC socket: {}", e))
        })?;

        Ok(Self { stream })
    }

    /// Send a message and wait for a response.
    pub async fn send_and_receive(&mut self, msg: IpcMessage) -> PhantomResult<IpcMessage> {
        let data = rmp_serde::to_vec(&msg).map_err(|e| {
            PhantomError::SerializationError(format!("Failed to serialize IPC message: {}", e))
        })?;

        // Write length prefix
        let len = data.len() as u32;
        self.stream.write_all(&len.to_le_bytes()).await.map_err(|e| PhantomError::IoError(e))?;
        
        // Write payload
        self.stream.write_all(&data).await.map_err(|e| PhantomError::IoError(e))?;

        // Read response length
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await.map_err(|e| PhantomError::IoError(e))?;
        let resp_len = u32::from_le_bytes(len_buf) as usize;

        // Read response payload
        let mut resp_buf = vec![0u8; resp_len];
        self.stream.read_exact(&mut resp_buf).await.map_err(|e| PhantomError::IoError(e))?;

        let resp_msg: IpcMessage = rmp_serde::from_slice(&resp_buf).map_err(|e| {
            PhantomError::SerializationError(format!("Failed to deserialize IPC message: {}", e))
        })?;

        Ok(resp_msg)
    }
}
