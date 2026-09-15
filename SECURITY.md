# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

PHANTOM takes the security of local model execution and runtime memory safety seriously.

If you discover a security vulnerability (such as a buffer overflow, out-of-bounds tensor read, arbitrary code execution via GGUF parsing, or local privilege escalation):

1. **Do not create a public GitHub issue.**
2. Send a detailed description of the vulnerability directly to the project maintainer via GitHub Private Vulnerability Reporting or email `security@phantom-runtime.local`.
3. Include:
   - Reproduction steps or proof-of-concept exploit script
   - Affected PHANTOM version and environment details
   - Impact assessment

## Security Practices

- **Zero-Arbitrary Code Execution**: PHANTOM strictly loads GGUF model tensors and avoids execution of pickled Python weights (`.bin` / `.pt`).
- **Memory Safety**: Direct memory mapping and GPU tensor allocations are bounds-checked to prevent buffer overruns and unauthenticated system memory reads.
