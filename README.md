# Auction Simulator App

A web-based token auction simulator built with Rust, Leptos, and WebAssembly. This application simulates token auctions with various market conditions, bidder profiles, and learning algorithms.

**Live Demo:** [https://surus1337.github.io/auctions/](https://surus1337.github.io/auctions/)

## Features

- Interactive auction simulation with real-time visualization
- Multiple market conditions (Bull, Bear, Neutral, Volatile)
- Dynamic bidder profiles with learning capabilities
- Detailed analytics and charts
- Export functionality for auction data
- Dark mode support

## Prerequisites

- Rust (latest stable version)
- Node.js and npm (for Trunk)
- A modern web browser

## Installation

1. Install Rust:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Install Trunk (Rust WASM web application bundler):
   ```bash
   cargo install trunk
   ```

3. Clone the repository:
   ```bash
   git clone https://github.com/surus1337/auctions.git
   cd auctions
   ```

4. Build and run the application:
   ```bash
   trunk serve
   ```

The application will be available at `http://localhost:8080`

## Usage

### Running Auctions

1. Configure the auction parameters in the Configuration tab:
   - Trust Assets
   - Redemption Percentage
   - Number of Bidders
   - Total Tokens
   - Growth Rates

2. Use the "Run Auction" button to start an auction
3. View results in real-time
4. Use "Next Auction" to proceed with growth parameters

### Exporting Data

The Analytics tab provides options to export data in various formats:
- JSON export of complete auction history
- CSV export of auction summaries
- CSV export of bidder performance

## Development

### Project Structure

- `src/main.rs`: Main application code
- `index.html`: Web entry point
- `Cargo.toml`: Rust dependencies and configuration

### Building for Production

```bash
trunk build --release
```

The production build will be available in the `dist` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is MIT licensed.

---

Vibe coded with love and MIT licensed.

Please leave any suggestions or feedback. All robot approved changes will be merged.

-pm
