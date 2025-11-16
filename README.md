# Multi-Label Bug Report Classification

A flexible, configurable pipeline for multi-label classification of software defect reports. Supports traditional ML and deep learning models, feature selection, and rich visualizations.

---

## Features
- **Configurable via JSON:** Easily control data, features, models, and output from a single config file.
- **Supports Multiple Models:** MultinomialNB, Logistic Regression, Random Forest, MLP, CNN.
- **Feature Engineering:** TF-IDF, Chi-Square selection, wordcloud-based vocabularies.
- **Balanced/Unbalanced Data:** Run experiments on both original and balanced datasets.
- **Visualizations:** Automatic generation of plots for data, features, and model results.
- **Output Organization:** All results and plots saved in organized folders by experiment name.

---

## Quick Start

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Your Experiment**
   - Edit or copy a config file in `configs/` (see `quick_test.json`).
   - See [`CONFIG_HELP.md`](CONFIG_HELP.md) for all config options and their impact.

3. **Run an Experiment**
   ```bash
   python src/configurable_main.py configs/quick_test.json
   ```
   - All outputs will be saved in `output/<experiment_name>/`.

4. **Review Results**
   - Open CSVs and PNGs in the output folder.
   - Use visualizations to compare models and understand data.

---

## Project Structure
```
├── configs/           # JSON config files for experiments
├── data/              # Input dataset(s)
├── output/            # All results and visualizations
├── src/               # Source code
│   ├── configurable_main.py  # Main experiment runner
│   └── utils/         # Modular utilities
├── requirements.txt   # Python dependencies
├── CONFIG_HELP.md     # Detailed config documentation
├── README.md          # This file
```

---

## Notebooks
- [Google Colab Notebook](multilable_prediction.ipynb) for interactive runs and demos.

---

## Dataset
- [Dataset CSV](dataset.csv) (see `data/` folder for local use)

---

## Advanced Usage
- Tune any parameter in your config file for custom experiments.
- Add new models or features by extending the code in `src/utils/`.
- For help on config options, see [`CONFIG_HELP.md`](CONFIG_HELP.md).

---

## License
MIT License

---

## Author
- Jalaj Pachouly ([GitHub](https://github.com/jalajpachouly))

---

## Support
For questions, open an issue or contact the author via GitHub.
