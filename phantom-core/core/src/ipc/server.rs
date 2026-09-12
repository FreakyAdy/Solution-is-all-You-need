//! # IPC Server Implementation
//!
//! Multi-client asynchronous IPC server communicating over Named Pipes (Windows)
//! or Unix Domain Sockets (Linux). Handles model loading, generation dispatch,
//! and live telemetry queries from Python and the API Gateway.

use crate::{
    engine::PhantomEngine,
    ipc::IpcMessage,
    PhantomError, PhantomResult,
};
use interprocess::local_socket::{
    tokio::{LocalSocketListener, LocalSocketStream},
    NameTypeSupport, ToNsName,
};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

pub struct IpcServer {
    socket_name: String,
    engine: Arc<RwLock<Option<PhantomEngine>>>,
}

impl IpcServer {
    pub fn new(socket_name: &str) -> Self {
        Self {
            socket_name: socket_name.to_string(),
            engine: Arc::new(RwLock::new(None)),
        }
    }

    pub fn with_engine(socket_name: &str, engine: PhantomEngine) -> Self {
        Self {
            socket_name: socket_name.to_string(),
            engine: Arc::new(RwLock::new(Some(engine))),
        }
    }

    /// Start listening and accepting connections.
    pub async fn run(&self) -> PhantomResult<()> {
        let name = self.socket_name.as_str().to_ns_name::<NameTypeSupport>().map_err(|e| {
            PhantomError::IpcError(format!("Invalid IPC socket name: {}", e))
        })?;

        let listener = LocalSocketListener::bind(name).map_err(|e| {
            PhantomError::IpcError(format!("Failed to bind to IPC socket {}: {}", self.socket_name, e))
        })?;

        info!(socket = %self.socket_name, "PHANTOM IPC Server listening for connections");

        loop {
            match listener.accept().await {
                Ok(stream) => {
                    let engine_ref = Arc::clone(&self.engine);
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(stream, engine_ref).await {
                            warn!("IPC client connection ended with error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Error accepting IPC client: {}", e);
                }
            }
        }
    }

    async fn handle_connection(
        mut stream: LocalSocketStream,
        engine_ref: Arc<RwLock<Option<PhantomEngine>>>,
    ) -> PhantomResult<()> {
        loop {
            // Read 4-byte message length
            let mut len_buf = [0u8; 4];
            match stream.read_exact(&mut len_buf).await {
                Ok(_) => {},
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(PhantomError::IoError(e)),
            };

            let msg_len = u32::from_le_bytes(len_buf) as usize;
            if msg_len > 16 * 1024 * 1024 {
                return Err(PhantomError::IpcError("Message size exceeded 16MB limit".into()));
            }

            let mut msg_buf = vec![0u8; msg_len];
            stream.read_exact(&mut msg_buf).await.map_err(PhantomError::IoError)?;

            let msg: IpcMessage = rmp_serde::from_slice(&msg_buf).map_err(|e| {
                PhantomError::SerializationError(format!("Failed to decode MessagePack IPC message: {}", e))
            })?;

            // Process message and generate response
            let response = match msg {
                IpcMessage::Status { .. } => {
                    let guard = engine_ref.read().await;
                    let ready = guard.is_some();
                    let mem = if let Some(eng) = guard.as_ref() {
                        let rep = eng.memory_report();
                        rep.vram_used_mb
                    } else {
                        0.0
                    };
                    IpcMessage::Status {
                        ready,
                        memory_used_mb: mem,
                    }
                }
                IpcMessage::LoadModel { model_path, profile_path } => {
                    info!(model = %model_path, profile = %profile_path, "IPC request to load model");
                    let mut guard = engine_ref.write().await;
                    let config = crate::EngineConfig::default();
                    match PhantomEngine::new(&model_path, &profile_path, config).await {
                        Ok(eng) => {
                            *guard = Some(eng);
                            IpcMessage::Status { ready: true, memory_used_mb: 512.0 }
                        }
                        Err(e) => IpcMessage::Error { message: e.to_string() },
                    }
                }
                IpcMessage::Generate { prompt, max_tokens, temp } => {
                    let guard = engine_ref.read().await;
                    if let Some(eng) = guard.as_ref() {
                        let mut params = crate::SamplingParams::default();
                        params.max_tokens = max_tokens;
                        params.temperature = temp;
                        match eng.generate(&prompt, params).await {
                            Ok(text) => IpcMessage::Token { text, is_final: true },
                            Err(e) => IpcMessage::Error { message: e.to_string() },
                        }
                    } else {
                        IpcMessage::Error { message: "No model is currently loaded in PHANTOM engine".into() }
                    }
                }
                IpcMessage::Token { .. } | IpcMessage::Error { .. } => {
                    IpcMessage::Error { message: "Invalid client-to-server IPC message".into() }
                }
            };

            // Write back response
            let resp_bytes = rmp_serde::to_vec(&response).map_err(|e| {
                PhantomError::SerializationError(format!("Failed to serialize IPC response: {}", e))
            })?;
            let resp_len = resp_bytes.len() as u32;
            stream.write_all(&resp_len.to_le_bytes()).await.map_err(PhantomError::IoError)?;
            stream.write_all(&resp_bytes).await.map_err(PhantomError::IoError)?;
            stream.flush().await.map_err(PhantomError::IoError)?;
        }

        Ok(())
    }
}
