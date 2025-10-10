# Scaler Wizard

Scaler Wizard is the implementation hub for Daniel Ramos' Multi-Vibe Code In Chain initiative to extend TransformerLab with accessible model scaling workflows. It provides a suite of tools and components to automate and manage the process of scaling up machine learning models, with a focus on safety, efficiency, and developer experience.

## Getting Started

Follow these instructions to get a local copy of the project up and running for development and testing purposes.

### Prerequisites

This project uses Python and Node.js. You will need to have them installed on your system.

*   Python (3.8+ recommended)
*   Node.js and npm (for UI components)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/scaler-wizard.git
    cd scaler-wizard
    ```

2.  **Set up the Python environment:**
    It is recommended to use a virtual environment.
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
    Install the required Python packages. While a `requirements.txt` is not yet present, the following packages are used by various components:
    ```sh
    pip install pyyaml gputil psutil jsonschema
    ```

3.  **Set up the UI environment (optional):**
    If you are working on the UI components, navigate to the `ui` directory and install the dependencies.
    ```sh
    cd ui
    npm install
    cd ..
    ```

## Usage

Scaler Wizard is a collection of components designed to be integrated into a larger model scaling workflow. Here are some of the key parts and how to use them:

*   **Scaling Engine (`components/scaling_engine`):** Provides adaptive heuristics to suggest LoRA configurations based on your hardware and scaling targets.
*   **Safety Circuit Breaker (`components/safety`):** Monitors training metrics and can automatically halt or warn if performance degrades, preventing wasted resources.
*   **Profile Recommender (`components/scaling`):** Detects your hardware and recommends a suitable scaling profile from a set of templates.
*   **Rollback Tool (`tools/rollback.py`):** A command-line script to revert a training run to a previous, known-good phase.
    ```sh
    python tools/rollback.py --run-dir /path/to/your/run --target-phase 1
    ```

## Project Structure

The repository is organized into the following main directories:

-   `components/`: Core Python modules for different aspects of the scaling workflow.
    -   `cache/`: Caching for training checkpoints.
    -   `distillation/`: Scaffolding for knowledge distillation.
    -   `evaluation/`: Utilities for progressive context evaluation.
    -   `safety/`: The circuit breaker and other safety components.
    -   `scaling/`: The hardware-aware profile recommender.
    -   `scaling_engine/`: The core adaptive scaling logic.
    -   `telemetry/`: A lightweight metrics logger.
    -   `validation/`: JSON schema validation helpers.
-   `docs/`: Project documentation, including architecture, process, and backlog.
-   `specs/`: Formal specifications for APIs, data schemas (`cockpit_schema.json`), and interfaces.
-   `tools/`: Standalone scripts for managing scaling runs, like `rollback.py`.
-   `ui/`: Frontend components for the "cockpit" dashboard.
    -   `services/`: Client-side services like the `voice_pipeline.js` for audio alerts.
-   `research/`: Research notes and comparisons of different techniques.
-   `benchmarks/`: Datasets and results for evaluating scaling performance.
-   `community/`: A place for shared patterns and community contributions.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the terms of the LICENSE file.