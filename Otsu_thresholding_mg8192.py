#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Mukhil Sundararaj Gowthaman
# @Date:   2025-02-21 10:45:00
# @Last Modified by:   Mukhil Sundararaj Gowthaman
# @Last Modified time: 2025-02-28 13:16:10

from PIL import Image
import matplotlib.pyplot as plt
import time
import pandas as pd
import os

def convert_to_grayscale(img):
    """
    Convert a color image to grayscale using the formula:
      I = Round(0.299*R + 0.587*G + 0.114*B)
    """
    width, height = img.size
    pixels = list(img.getdata())
    gray_pixels = []
    for pixel in pixels:
        r, g, b = pixel[:3]  # ignore alpha if present
        gray_val = round(0.299 * r + 0.587 * g + 0.114 * b)
        gray_pixels.append(gray_val)
    gray_img = Image.new("L", (width, height))
    gray_img.putdata(gray_pixels)
    return gray_img, gray_pixels

def compute_histogram(gray_pixels):
    """
    Compute the histogram for the grayscale image.
    Returns both the raw counts (for plotting) and the normalized histogram.
    """
    hist = [0] * 256
    for pixel in gray_pixels:
        hist[pixel] += 1
    return hist

def dynamic_otsu_multi_threshold(hist, num_regions):
    """
    Compute multi-thresholds using a dynamic programming approach based on Otsu's method.
    This function computes thresholds for 2, 3, or 4 regions.
    Returns a list of optimal thresholds and the corresponding total within-class variance.
    """
    L = len(hist)
    total = sum(hist)
    # Compute normalized probabilities for each intensity level.
    p = [h / total for h in hist]
    
    # Precompute cumulative sums:
    # P: cumulative probability, S: cumulative sum of intensities, Q: cumulative sum of squared intensities.
    P = [0] * L
    S = [0] * L
    Q = [0] * L
    P[0] = p[0]
    S[0] = 0 * p[0]
    Q[0] = 0 * 0 * p[0]
    for i in range(1, L):
        P[i] = P[i-1] + p[i]
        S[i] = S[i-1] + i * p[i]
        Q[i] = Q[i-1] + i * i * p[i]
    
    def cost(i, j):
        """Compute the within-class variance for the segment [i, j]."""
        if i == 0:
            w = P[j]
            s = S[j]
            q = Q[j]
        else:
            w = P[j] - P[i-1]
            s = S[j] - S[i-1]
            q = Q[j] - Q[i-1]
        if w == 0:
            return 0
        return q - (s * s / w)
    
    # dp[i][k] holds the minimum cost for partitioning bins 0..i into k segments.
    dp = [[float('inf')] * (num_regions + 1) for _ in range(L)]
    # backtrack[i][k] records the partition point.
    backtrack = [[-1] * (num_regions + 1) for _ in range(L)]
    
    # Base case: partitioning into 1 segment.
    for i in range(L):
        dp[i][1] = cost(0, i)
    
    # Fill dp for segments 2 to num_regions.
    for k in range(2, num_regions + 1):
        for i in range(k - 1, L):
            for j in range(k - 2, i):
                val = dp[j][k - 1] + cost(j + 1, i)
                if val < dp[i][k]:
                    dp[i][k] = val
                    backtrack[i][k] = j
    
    # Backtrack to find the optimal thresholds.
    thresholds = []
    k = num_regions
    i = L - 1
    while k > 1:
        j = backtrack[i][k]
        thresholds.append(j)
        i = j
        k -= 1
    thresholds.sort()
    best_total_variance = dp[L - 1][num_regions]
    return thresholds, best_total_variance

def segment_image(gray_pixels, width, height, thresholds, num_regions):
    """
    Segment the image based on the computed thresholds.
    Uses different grayscale values to differentiate regions:
      - For 2 regions: [0, 255]
      - For 3 regions: [0, 128, 255]
      - For 4 regions: [0, 85, 170, 255]
    """
    if num_regions == 2:
        region_vals = [0, 255]
    elif num_regions == 3:
        region_vals = [0, 128, 255]
    elif num_regions == 4:
        region_vals = [0, 85, 170, 255]
    
    segmented = []
    for pixel in gray_pixels:
        if num_regions == 2:
            region = 0 if pixel <= thresholds[0] else 1
        elif num_regions == 3:
            if pixel <= thresholds[0]:
                region = 0
            elif pixel <= thresholds[1]:
                region = 1
            else:
                region = 2
        elif num_regions == 4:
            if pixel <= thresholds[0]:
                region = 0
            elif pixel <= thresholds[1]:
                region = 1
            elif pixel <= thresholds[2]:
                region = 2
            else:
                region = 3
        segmented.append(region_vals[region])
    seg_img = Image.new("L", (width, height))
    seg_img.putdata(segmented)
    return seg_img

def plot_histogram(hist, thresholds, base, num_regions, output_dir):
    """
    Plot the image histogram with vertical red lines indicating the computed thresholds.
    Saves the plot as a PNG file in the specified output directory.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(range(256), hist, width=1.0, edgecolor='black')
    for t in thresholds:
        plt.axvline(x=t, color='red', linestyle='--', linewidth=2)
    plt.title(f"{base}: Histogram with {num_regions}-region Thresholds")
    plt.xlabel("Grayscale Value")
    plt.ylabel("Frequency")
    hist_filename = os.path.join(output_dir, f"{base}-{num_regions}regions-hist.png")
    plt.savefig(hist_filename)
    plt.close()

def process_image(filename):
    """
    Process an image by:
      1. Converting it to grayscale.
      2. Computing the histogram.
      3. For 2, 3, and 4 regions:
         - Determining optimal thresholds using dynamic programming.
         - Segmenting the image.
         - Plotting the histogram with threshold overlays.
      4. Reporting processing time and results.
    """
    print("Processing:", filename)
    start_time = time.time()
    
    # Create an output directory for this image.
    base = os.path.splitext(os.path.basename(filename))[0]
    out_dir = "Project 1/" + base + "_outputs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Read the image and get its dimensions.
    img = Image.open(filename)
    width, height = img.size
    
    # Convert to grayscale.
    gray_img, gray_pixels = convert_to_grayscale(img)
    gray_filename = os.path.join(out_dir, base + "-gray.bmp")
    gray_img.save(gray_filename)
    
    # Compute histogram.
    hist = compute_histogram(gray_pixels)
    
    results = {}
    for num_regions in [2, 3, 4]:
        print(f"Segmenting into {num_regions} regions...")
        thresholds, total_variance = dynamic_otsu_multi_threshold(hist, num_regions)
        seg_img = segment_image(gray_pixels, width, height, thresholds, num_regions)
        out_filename = os.path.join(out_dir, base + f"-{num_regions}regions-out.bmp")
        seg_img.save(out_filename)
        
        # Plot and save the histogram with threshold overlays.
        plot_histogram(hist, thresholds, base, num_regions, out_dir)
        
        results[num_regions] = {"thresholds": thresholds, "total_variance": total_variance}
        print(f"  Thresholds: {thresholds}, Total Variance: {total_variance}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time for {filename}: {processing_time:.2f} seconds\n")
    return results, processing_time

def main():
    """
    Main execution block:
      - Processes a list of BMP images.
      - Generates segmentation outputs, histogram plots, and logs performance.
      - Creates a summary report saved as a CSV file.
    """
    image_files = ["Project 1/basket_balls.bmp", "Project 1/data13.bmp", "Project 1/tiger1.bmp"]
    summary_data = []
    
    for filename in image_files:
        results, proc_time = process_image(filename)
        for num_regions in results:
            summary_data.append({
                "Image": filename,
                "Regions": num_regions,
                "Thresholds": results[num_regions]["thresholds"],
                "Total Variance": results[num_regions]["total_variance"],
                "Processing Time (s)": round(proc_time, 2)
            })
    
    # Create and display a summary report.
    df = pd.DataFrame(summary_data)
    print("Summary Report:")
    print(df)
    if not os.path.exists("Project 1/summary"):
        os.makedirs("Project 1/summary")
    df.to_csv("Project 1/summary/segmentation_summary.csv", index=False)

if __name__ == "__main__":
    main()