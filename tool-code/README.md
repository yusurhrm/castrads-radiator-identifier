# Radiator Identification System

This is a FastAPI-based radiator/product identification system. The system is divided into a user-facing interface and an admin interface:

* The user interface guides users through a step-by-step decision flow to input radiator dimensions and characteristics. Users can upload one or multiple images at any stage of the process, allowing image understanding functionality to suggest feature values for later steps.
* The admin interface is responsible for uploading Excel datasets, configuring dimensions, training decision tree models, viewing model performance, configuring flow-step UI settings, configuring image-understanding fields, and switching the active model used by the user interface.

---

# Current Features

* Excel dataset upload and automatic dimension recognition
* Configuration of:

  * Dimension names
  * Display names
  * Data types
  * Feature weights
  * Default enabled states
* Decision tree model training, retraining, deletion, and activation
* Independent saving of training outputs into `models/<model_id>/`
* Accuracy, Coverage, and unmatched prediction analysis
* Decision flowchart visualisation for any trained model
* Configurable input types, guidance images, help text, and image-understanding participation for each flow step
* User-side flow input, back navigation, skipping, and early termination when confidence threshold is reached
* Floating image-understanding access within the user identification flow
* Multi-image upload, image preprocessing, Qwen image-understanding streaming responses, and result confirmation forms
* Image-understanding results only populate unanswered steps while preserving existing user input
* Detailed image-understanding logs written to `logs/vision_debug.log`

---

# Technology Stack

* Backend: FastAPI + Uvicorn
* Templates: Jinja2
* Data Processing: Pandas + OpenPyXL
* Image Processing: Pillow
* Image Understanding: OpenAI SDK compatible mode + DashScope/Qwen
* Model: Information-gain-based Decision Tree
* Dependency Management: uv

---

# Quick Start

## Install Dependencies

```bash
uv sync
```

## Start the Server

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

## Access Pages

```text
User Interface:
http://127.0.0.1:8000

Admin Interface:
http://127.0.0.1:8000/admin
```

If port `8000` is already in use:

```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

# Project Structure

```text
.
├── main.py                    # FastAPI application setup and route registration
├── admin_routes.py            # Admin routes: training, retraining, model management, flow configuration
├── user_routes.py             # User flow routes: homepage, start, answer, skip, back
├── vision_routes.py           # Image-understanding routes
├── flowchart_routes.py        # Decision flowchart routes
├── flow_views.py              # User-side next-step and result-page rendering
├── web_templates.py           # Jinja2 template engine
├── vision_workflow.py         # Image-understanding workflow and SSE handling
├── app_config.py              # Paths, runtime config, Qwen config, JSON utilities
├── app_logging.py             # Image-understanding logging
├── decision_tree.py           # Dataset loading, feature inference, decision tree training
├── dimension_defaults.py      # Default dimension configuration
├── flow_runtime.py            # User session state and runtime identification flow
├── model_store.py             # Model saving, loading, activation, retraining
├── vision_image_limits.py     # Image size/count validation and compression
├── vision_service.py          # Qwen prompts and image-understanding calls
├── templates/                 # HTML templates
├── static/                    # Static assets
├── models/                    # Runtime model directory
├── uploads/                   # Runtime upload directory
└── logs/                      # Runtime logs
```

---

# Dataset Format

The uploaded Excel dataset should follow a structure similar to `output.xlsx`.

* The first column is used as the target classification field (e.g. Product ID or SKU).
* Remaining columns are used as identification dimensions.
* Missing values are filled with `MISSING`.
* If the dataset contains:

  * `Plain`
  * `Flat top`
  * `Scroll`
  * `Round top`

  the system automatically derives a `Top Style` field.

Example:

| Products ID | Name      | Castrads SKU | Section Length (mm) | Leg Section Depth (mm) |
| ----------- | --------- | ------------ | ------------------- | ---------------------- |
| 637         | Product A | SKU-A        | 80                  | 165                    |
| 640         | Product B | SKU-B        | 100                 | 185                    |

---

# Admin Workflow

1. Open:

   ```text
   http://127.0.0.1:8000/admin
   ```

2. Upload an Excel dataset.

3. Configure dimensions before training:

   * `Use`: Include in training or not
   * `Name / Display Name`
   * `Type`: Numeric or Categorical
   * `Weight`: Manual feature weighting used in:

     ```text
     information_gain * weight
     ```
   * `Ease`: Measurement difficulty
   * `Measurement Comment`
   * `Image Description`

4. Train the model.

5. View:

   * Model details
   * Retraining
   * Activation
   * Flow configuration
   * Flowchart visualisation

---

# Decision Tree Logic

The model selects the optimal split dimension from enabled features.

## Categorical Features

* Split by unique values
* Evaluated using information gain

## Numeric Features

* Candidate thresholds are generated between neighbouring values
* Highest information-gain threshold is selected
* Produces:

  ```text
  <= threshold
  > threshold
  ```

Final split score:

```text
score = information_gain * weight
```

Feature weights are manually configured admin preferences, not confidence values.

---

# Accuracy & Coverage

After training, the system generates `metrics.json`.

* `Accuracy`:
  Percentage of correctly predicted samples among successfully classified samples.

* `Coverage`:
  Percentage of dataset rows that reach a valid prediction path.

* Unmatched prediction analysis records rows that could not be confidently classified.

Current metrics are based on training-set evaluation only and are not equivalent to independent test-set performance.

---

# User Identification Flow

After clicking `Start Identification`, the user enters the dynamic decision flow.

Each step displays:

* Current model
* Current confidence
* Remaining candidate count
* Current question
* Guidance images and help text
* Input controls
* Image-understanding suggestions
* Back / Skip / Submit controls

Input types are configurable through the admin panel:

* Auto
* Number
* Text
* Select

When the confidence threshold is reached, the flow ends early and displays the predicted radiator.

---

# Skip Logic

When a user clicks `Skip`:

* The candidate set is not filtered
* The dimension is marked as skipped
* The system avoids re-asking that dimension
* The next available dimension is selected dynamically

---

# Image Understanding

The image-understanding interface is accessed through a floating button within the user flow.

Features include:

* Single or multi-image upload
* Incremental image uploads
* Image preview
* Upload cancellation and return to flow
* Streaming inference loading page using SSE
* Editable result confirmation forms

After confirmation:

* Unanswered fields receive suggested values
* Existing user input is preserved
* Suggested values are still shown for comparison

---

# Qwen Configuration

Configured in `app_config.py`:

```python
QWEN_API_KEY = ""
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_VISION_MODEL = "qwen3.6-plus"
```

Environment variables may also be used:

```text
DASHSCOPE_API_KEY
QWEN_API_KEY
```

---

# Runtime Files

The following are generated at runtime and are normally excluded from version control:

* `.venv/`
* `active_model.json`
* `models/`
* `uploads/`
* `static/model_assets/`
* `logs/`
* `dimension_defaults.json`

When sharing the project, typically only the source code, templates, static default assets, `pyproject.toml`, `uv.lock`, and `README.md` are required.

---

# Known Limitations

* User sessions and image-understanding tasks are currently stored in server memory and are better suited for local demonstration environments.
* Model evaluation currently uses training-set evaluation only; future work could include train/validation splitting.
* Image understanding depends on external Qwen services and performance varies based on image count, size, network speed, and API response time.
* API keys may currently exist within configuration files, though production deployment should use environment variables or secure key management solutions.
