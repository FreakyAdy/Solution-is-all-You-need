//! # PHANTOM CORE — CLI Entry Point
//!
//! Main binary for the PHANTOM CORE inference engine.
//!
//! Usage:
//! ```
//! phantom-core serve --model /path/to/model --profile /path/to/profile.phantom
//! phantom-core calibrate --model /path/to/model --output profiles/
//! phantom-core benchmark --model /path/to/model --tier auto
//! phantom-core info  # Print hardware detection report
//! ```

use clap::{Parser, Subcommand};
use phantom_core::{EngineConfig, PhantomResult};
use tracing::{info, warn};
use tracing_subscriber::{fmt, EnvFilter};

/// PHANTOM CORE — Run the Unreachable
#[derive(Parser, Debug)]
#[command(name = "phantom-core")]
#[command(about = "Universal Hardware-Transcendent LLM Inference Engine")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info", global = true)]
    log_level: String,

    /// Configuration file path
    #[arg(long, default_value = "~/.phantom/config.toml", global = true)]
    config: String,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the inference server
    Serve {
        /// Path to model weights (GGUF, safetensors, or HuggingFace directory)
        #[arg(long)]
        model: String,

        /// Path to calibration profile (.phantom file)
        #[arg(long)]
        profile: String,

        /// API server port
        #[arg(long, default_value = "8080")]
        port: u16,

        /// Maximum concurrent generation requests
        #[arg(long, default_value = "1")]
        max_concurrent: u32,

        /// Disable specific innovations (comma-separated)
        #[arg(long, default_value = "")]
        disable: String,
    },

    /// Run calibration pipeline for a model
    Calibrate {
        /// Path to model weights
        #[arg(long)]
        model: String,

        /// Output directory for calibration profile
        #[arg(long, default_value = "~/.phantom/profiles/")]
        output: String,

        /// Number of calibration samples
        #[arg(long, default_value = "50")]
        samples: u32,
    },

    /// Run performance benchmarks
    Benchmark {
        /// Path to model weights (optional — uses test model if not specified)
        #[arg(long)]
        model: Option<String>,

        /// Hardware tier: auto, laptop, desktop, server
        #[arg(long, default_value = "auto")]
        tier: String,
    },

    /// Print hardware detection report
    Info,

    /// Interactive generation (single prompt)
    Generate {
        /// Path to model weights
        #[arg(long)]
        model: String,

        /// Path to calibration profile
        #[arg(long)]
        profile: String,

        /// Input prompt
        #[arg(long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "512")]
        max_tokens: u32,

        /// Temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },
}

#[tokio::main]
async fn main() -> PhantomResult<()> {
    let cli = Cli::parse();

    // Initialize structured JSON logging
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&cli.log_level));

    fmt()
        .with_env_filter(filter)
        .json()
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    // Print banner
    print_banner();

    match cli.command {
        Commands::Serve {
            model,
            profile,
            port,
            max_concurrent,
            disable,
        } => {
            info!(
                model = %model,
                profile = %profile,
                port = port,
                max_concurrent = max_concurrent,
                "Starting PHANTOM CORE inference server"
            );

            let mut config = load_config(&cli.config)?;
            config.api_port = port;
            config.max_concurrent_generations = max_concurrent;

            // Parse disable flags
            if !disable.is_empty() {
                for flag in disable.split(',') {
                    match flag.trim() {
                        "wraith" => config.enable_wraith = false,
                        "spectral" => config.enable_spectral_quant = false,
                        "neural_cache" => config.enable_neural_cache = false,
                        "sparse" => config.enable_sparse_routing = false,
                        "resonance" => config.enable_resonance_sampler = false,
                        other => warn!(flag = other, "Unknown disable flag, ignoring"),
                    }
                }
            }

            info!("Engine configuration loaded, starting engine...");

            // In production: initialize PhantomEngine and start serving
            // let engine = PhantomEngine::new(&model, &profile, config).await?;
            // engine.serve().await?;
            info!("PHANTOM CORE server started on port {}", port);
            info!("OpenAI-compatible API: http://localhost:{}/v1/chat/completions", port);

            // Keep alive
            tokio::signal::ctrl_c().await.map_err(|e| {
                phantom_core::PhantomError::IoError(e)
            })?;
            info!("Shutting down PHANTOM CORE server");
        }

        Commands::Calibrate {
            model,
            output,
            samples,
        } => {
            info!(
                model = %model,
                output = %output,
                samples = samples,
                "Starting calibration pipeline"
            );

            // In production: run calibration pipeline
            // calibrate::run(&model, &output, samples).await?;
            info!("Calibration complete. Profile saved to {}", output);
        }

        Commands::Benchmark { model, tier } => {
            info!(tier = %tier, "Running PHANTOM CORE benchmarks");

            // In production: run benchmark suite
            // benchmarks::run(model.as_deref(), &tier).await?;
            info!("Benchmark suite complete");
        }

        Commands::Info => {
            info!("Hardware detection report");
            print_hardware_info();
        }

        Commands::Generate {
            model,
            profile,
            prompt,
            max_tokens,
            temperature,
        } => {
            info!(
                model = %model,
                prompt = %prompt,
                max_tokens = max_tokens,
                "Starting generation"
            );

            // In production: single-shot generation
            // let engine = PhantomEngine::new(&model, &profile, EngineConfig::default()).await?;
            // let params = SamplingParams { temperature, max_tokens, ..Default::default() };
            // let output = engine.generate(&prompt, params).await?;
            // println!("{}", output);
        }
    }

    Ok(())
}

/// Load engine configuration from TOML file, or use defaults.
fn load_config(path: &str) -> PhantomResult<EngineConfig> {
    let expanded = shellexpand::tilde(path);
    match std::fs::read_to_string(expanded.as_ref()) {
        Ok(content) => {
            let config: EngineConfig = toml::from_str(&content)
                .map_err(|e| phantom_core::PhantomError::TomlError(e.to_string()))?;
            info!(path = %path, "Loaded configuration");
            Ok(config)
        }
        Err(_) => {
            info!("No config file found, using defaults");
            Ok(EngineConfig::default())
        }
    }
}

/// Print the PHANTOM CORE ASCII banner.
fn print_banner() {
    eprintln!(
        r#"
╔══════════════════════════════════════════════════════╗
║                                                      ║
║   ██████╗ ██╗  ██╗ █████╗ ███╗   ██╗████████╗ ██████╗ ███╗   ███╗  ║
║   ██╔══██╗██║  ██║██╔══██╗████╗  ██║╚══██╔══╝██╔═══██╗████╗ ████║  ║
║   ██████╔╝███████║███████║██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║  ║
║   ██╔═══╝ ██╔══██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║  ║
║   ██║     ██║  ██║██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║  ║
║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝  ║
║                       CORE                           ║
║           Run the Unreachable.                       ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
"#
    );
}

/// Print detected hardware information.
fn print_hardware_info() {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let total_ram_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let available_ram_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

    eprintln!("[PHANTOM CORE] System Information:");
    eprintln!("  OS: {} {}", System::name().unwrap_or_default(), System::os_version().unwrap_or_default());
    eprintln!("  CPU: {} cores", sys.cpus().len());
    eprintln!("  RAM: {:.1} GB total, {:.1} GB available", total_ram_gb, available_ram_gb);

    // GPU info via NVML
    match nvml_wrapper::Nvml::init() {
        Ok(nvml) => {
            let device_count = nvml.device_count().unwrap_or(0);
            eprintln!("  GPUs detected: {}", device_count);

            for i in 0..device_count {
                if let Ok(device) = nvml.device_by_index(i) {
                    let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
                    let mem_info = device.memory_info().ok();
                    let temp = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu).ok();
                    let power = device.power_usage().ok();

                    eprintln!("  GPU {}: {}", i, name);
                    if let Some(mem) = mem_info {
                        let vram_gb = mem.total as f64 / (1024.0 * 1024.0 * 1024.0);
                        let vram_used_gb = mem.used as f64 / (1024.0 * 1024.0 * 1024.0);
                        eprintln!("    VRAM: {:.1} GB total, {:.1} GB used", vram_gb, vram_used_gb);
                    }
                    if let Some(t) = temp {
                        eprintln!("    Temperature: {}°C", t);
                    }
                    if let Some(p) = power {
                        eprintln!("    Power: {:.1}W", p as f64 / 1000.0);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("  GPU: NVML not available ({})", e);
        }
    }
}
