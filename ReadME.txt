Handwritten Letter Classification Project

Python Version:
- 3.10.0

How to Run the App:
To launch the Streamlit application, open your terminal and run:
streamlit run app.py

The dataset archive contains images you can upload to test the model’s predictions.

---

Known Issues & Recommendations:

- TensorFlow Installation:
  After installing TensorFlow, you might face compatibility issues with numpy.
  Solution: Install a compatible numpy version by running:
  pip install "numpy<2"

- Protobuf Compatibility:
  TensorFlow and Streamlit sometimes have conflicts with the protobuf package.
  Solution: If you encounter protobuf-related errors, try uninstalling protobuf:
  pip uninstall protobuf
  Reinstall only if necessary.

- VS Code File Location Issue:
  If VS Code cannot locate files when running the app, try these steps:
  - Open your project folder in VS Code by Shift + Right Click on the folder, then select Open with Code.
  - Alternatively, you can use Jupyter Notebook or open the folder via a PowerShell window.

---
