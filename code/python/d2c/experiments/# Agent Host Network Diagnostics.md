# Agent Host Network Diagnostics

## Local OS Proxy Configuration (@vscode/os-proxy-resolver)

- Proxy environment: (none)
- Auto-detect: true
- DHCP WPAD: unsupported
- DNS WPAD: not-found
- Configured PAC: unconfigured
- PAC: (none)
- Static rules: (none)
- Platform settings: Linux, mode=auto, ignored hosts=(none)

- Connections: 1 (1 local, 0 remote)

Connectivity probes run inside each agent host process (local or remote), so results reflect the environment the Copilot SDK actually connects from.

## Local agent host

- Agent host version: 1.135.0
- OS: linux (x64)
- Account: johnsnowman488-oss
- Proxy settings: (none)
- Proxy environment: (none)

### GitHub API

- URL: https://api.github.com
- DNS IPv4: 140.82.121.5 (10 ms)
- DNS IPv6: error (35 ms): getaddrinfo ENOTFOUND api.github.com
- Proxy: None
- Local OS proxy (@vscode/os-proxy-resolver): direct
- Reachability: ✓ 200 via direct (514 ms)

### Copilot API (CAPI)

- URL: https://api.individual.githubcopilot.com/_ping
- DNS IPv4: 140.82.113.21 (97 ms)
- DNS IPv6: error (342 ms): getaddrinfo ENOTFOUND api.individual.githubcopilot.com
- Proxy: None
- Local OS proxy (@vscode/os-proxy-resolver): direct
- Reachability: ✓ 200 via direct (1180 ms)

