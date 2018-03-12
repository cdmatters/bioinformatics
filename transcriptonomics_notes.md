# Transcriptonomics

## Overview

* To understand a system as complex as a cell need a way of characterising/collecting as much data as possible
* Therefore need to measure gene expression in its fullest capacity
* Transcriptomics uses the MicroArray techniques to measure expression of all 30K human genes
* This is more difficult for proteomics (not all soluble) and metabolomics.
* Above practical difficulties aside, much of transcriptomics data is transferrable to other -omics data

### What is Transcriptomics (again?)

Its about the quantitative measurement of how much mRNA is being expressed at each point
This is useful because it can give you insight into what's happening in the cell.

*CF Khan et al (2001) - classify different tyes of small round blue cells using only expression data (w neural networks). 
Vital classifications for treatments etc. Note, used PCA for initial dimensionality reduction (transcription factors 
regulate 100s of genes, so necessary to reduce to latent factors!)*

### Measuring Transcriptomics

#### 1. Spotted Microarrays
    * since mRNA is sensitive to degradation, it is converted to cDNA by reverse transcription
    * in this process fluorescent indicators are added (to the nucleotides - a different one for each sample)
    * a glass slide with an array of 'spots', each spot containing a complement to a specific, known mRNA sequence
    * when the matching samples recombine with the known sequences (hybridisation), the fluorescence is emitted
    * the microarray is then scanned/photographed - strength of fluorescence indicating the concentration of mRNA

Always comparing different samples, different colours (wavelengths) are used for each sample
  * A = green (cy3), B = red (cy5)
  * A > B -> green; A < B -> red; A == B -> yellow (both fire); None -> black
  
If comparing more than 2 samples, prepare a Common Reference cDNA Sample (which is a norm/standard/control). 
Then can just need to prepare A-CRS, B-CRS, C-CRS, and get 6 comparisons for free (etc)

#### 2. Affymetrix GeneChip Manufacture
  * get all cDNAs and label as before
  * but build up on a silicon chip nucleotide by nucleotide in 25 sequence bits
  * this allows great density of genes on a chip, and guarantee to

Comparison: Affymetrix vs Spotted
    1. Spotted is pairwise only, since variation in spots shapes & sizes vary a lot, affymetrix is just one sample
    2. Affymetrix is so precise and uniform, that v few variations, spotted is variable  (hence pairwise) 
    3. Spotted does whole gene generally, Affymetrix 25 nucleotides tops.
    3. (Affymetrix replicable/v similar, Spotted not)

Affymetrix preferred in industry.


