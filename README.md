# Halftone Color: Space-Filling Curve Halftoning for RGB Images

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.5+-brightgreen)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<img src="Images/halftone_color.png" alt="Pipeline of the color mode halftone" width="800">

An implementation of color halftoning using space-filling curves, extending the original monochrome algorithm from the SIGGRAPH '91 paper.

## Table of Contents
- [About](#about)
- [Algorithm](#algorithm)
- [Effects](#algorithm)
- [Examples](#examples)
- [Installation](#installation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## About

This project implements a color version of the halftoning algorithm described in:

**"Digital halftoning with space filling curves"**  
*Luiz Velho and Jonas de Miranda Gomes*  
SIGGRAPH '91 ([DOI: 10.1145/122718.122727](https://doi.org/10.1145/122718.122727))

The monochrome version was reimplemented during the 2025 summer course [Reproducing Results in Computer Graphics](https://lhf.impa.br/cursos/rr/)by me, Bianca Zavadisk de Abreu, Gustavo Souza Cardoso, Igor Augusto Zwirtes, Igor Roberto Alves, João Marcelo, Lucas Barros Barcelos, Pedro Henrique Porto and Yan Hill at the Instituto de Mathematica Pura e Aplicada (IMPA). This extension was made by me and adds a RGB image support to the original monochrome implementation. The monochrome version was reimplemented. We used Python because it had compatibility with OpenCV and because it was the most known programming language by the group.

### Requirements
- Python 3.8+
- OpenCV
- NumPy

## Algorithm

The halftone works with 3 core components: Hilbert Curve Generation, Curve Generation and the Halftoning Engine

#### 1. Space Filling Curve Generation (`hilbert()/peano()/lebesgue()`)

Maps a 1D index to 2D coordinates on the specified space filling curve

#### 2. Curve Generation (generate_space_filling_curve()`)

Calculates minimum curve order to cover image dimensions, and then generates complete curve coordinates
The space-filling curve approach provides:
- Superior dot distribution compared to regular grids
- Better detail preservation
- Smother tonal transitions

#### 3. Halftoning Engine (`halftoning()`)

The color halftone implementation:
1. Decomposes the RGB image into three separated channels, one for each color.
2. Applies the space-filling curve halftoning to each channel
3. Then, recombines channels with color correction
4. Uses optional pre-processing for enhanced results

## Effects

### Brightness Adjustment

Adjusts image brightness using a linear transformation formula:

$ g(x) = \alpha f(x) + \beta $

Where $\alpha > 0$, $\Beta$ is the gain(contrast) and bias(brightness) parameters.

## Installation

### Setup
```bash
git clone https://github.com/yourusername/halftone-color.git
cd halftone-color
pip install -r requirements.txt