# Custom Street Flood Index (CSFI)

**CSFI** is a tool designed to **detect flooded streets** using satellite images.
It takes two optical remote sensing scenes—**before** and **after** a flood—and outputs a prediction showing which streets are flooded.

Our goal is to make this tool **accessible and easy to use**, even for people who are not remote sensing experts.

<p align="center">
  <img src="figure.png" alt="Example of CSFI output"/>
</p>

---

## Quick Start

### 1. Download the Code

* Clone this repository with Git:

  ```bash
  git clone https://github.com/quentin-euler/CSFI.git
  ```

  or download the ZIP file and unzip it.

### 2. Install the Required Packages

In your terminal, go to the project folder and run:

```bash
pip install .
```

This will install all the Python libraries you need.
**Note:** The code was tested with **Python 3.12.10**.

---

## Run the Model

Once everything is installed, run:

```bash
python src/main.py
```

### What you need to provide:

In the main.py file you will need to modify the path to your two images :
* **A scene *after* the flood** (optical image of the flooded area)
* **A scene *before* the flood** (preferably from the same month in a different year to reduce seasonal bias)

You can use either:

* **TIF files** (recommended if you already pre-processed the data), or
* **.SAFE files** (Sentinel-2 products from Copernicus)

---

## Input Data Format

### If you use **TIF files**:

* The TIF must have **4 bands**: **Blue**, **Green**, **Red**, and **NIR** (Near-Infrared)
* **Reflectance values** should be **between 0 and 1**
* If you want to exclude clouds, set clouded pixels to **NaN**

### If you use **.SAFE files** (Sentinel-2):

* You **must provide an area of interest (extent)**, otherwise the street extraction can be **very slow**.
* The extent should be in **GeoJSON** format.
  Example:

```json
{
  "type": "Polygon",
  "coordinates": [
    [
      [-51.097183, -29.964949],
      [-51.250076, -29.964949],
      [-51.250076, -30.09474],
      [-51.097183, -30.09474],
      [-51.097183, -29.964949]
    ]
  ]
}
```

(Use **latitude/longitude** coordinates)

---

## Visualize the Results

Once the model has run, you can explore the predictions with the notebook:

```bash
src/visualize_results.ipynb
```

This will let you see which streets were detected as flooded.


