import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter

def detect_mt_from_residuals(tis, signal, fitted, threshold=0.5):
    """
    Detect MT effects from fitting residuals
    
    Parameters:
    -----------
    tis : array
        Inversion times
    signal : array
        Measured signal
    fitted : array
        Fitted signal
    threshold : float
        Correlation threshold for MT detection
    
    Returns:
    --------
    mt_detected : bool
        Whether MT effects were detected
    correlation : float
        Correlation coefficient
    p_value : float
        Statistical p-value
    """
    residuals = signal - fitted
    
    # Look for systematic residual patterns
    # MT typically causes oscillating residuals
    if len(tis) > 3:
        corr, p_val = pearsonr(tis[:-1], residuals[:-1])
        mt_detected = abs(corr) > threshold and p_val < 0.05
    else:
        corr, p_val = 0.0, 1.0
        mt_detected = False
    
    return mt_detected, corr, p_val

def estimate_mt_fraction(tis, residuals, signal=None, method='empirical'):
    """
    Estimate MT fraction from residual patterns
    
    Parameters:
    -----------
    tis : array
        Inversion times
    residuals : array
        Fitting residuals
    signal : array, optional
        Original signal values
    method : str
        Estimation method ('empirical' or 'enhanced')
    
    Returns:
    --------
    mt_fraction : float
        Estimated MT pool fraction
    k_exchange : float
        Exchange rate constant
    """
    # Simple empirical estimation
    residual_amplitude = np.std(residuals)
    
    if method == 'enhanced' and signal is not None:
        # Enhanced method using signal amplitude
        signal_amplitude = np.mean(np.abs(signal))
        # Empirical relationship with signal consideration
        mt_fraction = 0.15 * (residual_amplitude / signal_amplitude)
    else:
        # Basic method
        signal_amplitude = np.mean(np.abs(residuals))
        mt_fraction = 0.1 * (residual_amplitude / signal_amplitude)
    
    mt_fraction = np.clip(mt_fraction, 0.0, 0.3)  # Typical range 0-30%
    
    # Estimate exchange rate (simplified)
    k_exchange = 20.0  # Hz, typical for tissue
    
    return mt_fraction, k_exchange

def calculate_mt_correction_factor(mt_fraction, k_exchange, tis, tr, flip_angle, 
                                 tissue_type='generic', field_strength=1.5):
    """
    Calculate T1 correction factor due to MT effects
    
    Parameters:
    -----------
    mt_fraction : float
        MT pool fraction
    k_exchange : float
        Exchange rate
    tis : array
        Inversion times
    tr : float
        Repetition time
    flip_angle : float
        Flip angle in degrees
    tissue_type : str
        Type of tissue ('generic', 'cardiac', 'brain')
    field_strength : float
        MRI field strength in Tesla
    
    Returns:
    --------
    correction_factor : float
        Multiplicative correction factor for T1
    """
    if mt_fraction <= 0:
        return 1.0
    
    # Field strength dependent correction
    if field_strength <= 1.5:
        base_correction = 0.3  # Lower MT effect at 1.5T
    else:
        base_correction = 0.5  # Higher MT effect at 3T
    
    # Tissue-specific adjustments
    tissue_factors = {
        'cardiac': 1.2,  # Higher MT in cardiac tissue
        'brain': 1.1,
        'generic': 1.0
    }
    tissue_factor = tissue_factors.get(tissue_type, 1.0)
    
    # Calculate correction factor
    correction_factor = 1.0 + mt_fraction * base_correction * tissue_factor
    
    # Ensure reasonable bounds
    correction_factor = np.clip(correction_factor, 1.0, 1.5)
    
    return correction_factor

def identify_septal_region(t1_map, percentile_threshold=15):
    """
    Identify the septal region in the T1 map based on T1 values
    
    Parameters:
    -----------
    t1_map : array
        T1 map
    percentile_threshold : float
        Percentile threshold for septal identification
    
    Returns:
    --------
    septal_mask : array
        Boolean mask identifying septal regions
    """
    import numpy as np
    
    # Get valid T1 values
    valid_mask = ~np.isnan(t1_map)
    if not np.any(valid_mask):
        return np.zeros_like(t1_map, dtype=bool)
    
    valid_t1 = t1_map[valid_mask]
    
    # Use percentile to identify potential septal regions
    # Septum typically has more homogeneous and lower T1 values
    lower_thresh = np.percentile(valid_t1, percentile_threshold)
    upper_thresh = np.percentile(valid_t1, 50)  # Upper bound to avoid including blood
    
    # Create initial mask based on T1 range
    septal_mask = (t1_map >= lower_thresh) & (t1_map <= upper_thresh)
    
    # Add additional spatial constraints if needed
    # This is a simplified approach - a more robust method would use
    # connected components and spatial information
    
    return septal_mask

def apply_mt_correction_to_t1map(t1_map, mt_map=None, uniform_correction=None, septal_mask=None, septal_boost=None):
    """
    Apply MT correction to T1 map with pixel-wise variation
    
    Parameters:
    -----------
    t1_map : array
        Uncorrected T1 map
    mt_map : array, optional
        Pixel-wise MT correction factors
    uniform_correction : float, optional
        Uniform correction factor
    septal_mask : array, optional
        Mask identifying septal regions
    septal_boost : float, optional
        Additional correction factor for septal regions
    
    Returns:
    --------
    t1_corrected : array
        MT-corrected T1 map
    """
    import numpy as np
    
    # Make a copy to avoid modifying the original
    t1_corrected = t1_map.copy()
    
    # Track which pixels were modified
    modified_mask = np.zeros_like(t1_map, dtype=bool)
    
    # Apply pixel-wise correction if provided
    if mt_map is not None:
        # Calculate the correction factors: 1.0 + mt_fraction
        correction_factors = 1.0 + mt_map
        
        # Only apply to valid pixels in both maps
        valid_mask = ~np.isnan(t1_map) & ~np.isnan(mt_map) & (mt_map > 0)
        if np.any(valid_mask):
            t1_corrected[valid_mask] = t1_map[valid_mask] * correction_factors[valid_mask]
            modified_mask[valid_mask] = True
            
            print(f"Applied pixel-wise MT correction to {np.sum(valid_mask)} pixels")
            print(f"Mean correction factor: {np.mean(correction_factors[valid_mask]):.3f}")
    
    # Apply uniform correction if provided and not already corrected
    elif uniform_correction is not None:
        # Apply to all valid T1 pixels that haven't been corrected yet
        valid_mask = ~np.isnan(t1_map) & ~modified_mask
        if np.any(valid_mask):
            t1_corrected[valid_mask] = t1_map[valid_mask] * uniform_correction
            modified_mask[valid_mask] = True
            
            print(f"Applied uniform MT correction ({uniform_correction:.3f}) to {np.sum(valid_mask)} pixels")
    
    # Apply septal boost if provided
    if septal_mask is not None and septal_boost is not None:
        # Only apply to valid septal pixels
        septal_valid = septal_mask & ~np.isnan(t1_corrected)
        if np.any(septal_valid):
            t1_corrected[septal_valid] = t1_corrected[septal_valid] * septal_boost
            
            print(f"Applied septal boost ({septal_boost:.3f}) to {np.sum(septal_valid)} pixels")
    
    return t1_corrected

def identify_septal_region(t1_map, percentile_threshold=15):
    """
    Identify septal region based on T1 values
    
    Parameters:
    -----------
    t1_map : array
        T1 map
    percentile_threshold : float
        Percentile threshold for identifying septal region
    
    Returns:
    --------
    septal_mask : array
        Boolean mask identifying septal region
    """
    # Get valid T1 values
    valid_t1 = t1_map[~np.isnan(t1_map)]
    
    if len(valid_t1) == 0:
        return np.zeros_like(t1_map, dtype=bool)
    
    # Calculate threshold based on percentile
    threshold = np.percentile(valid_t1, percentile_threshold)
    
    # Create mask for potential septal region
    septal_mask = (t1_map > threshold) & ~np.isnan(t1_map)
    
    # Optional: Apply morphological operations to clean up the mask
    from scipy.ndimage import binary_opening, binary_closing
    septal_mask = binary_opening(septal_mask, iterations=1)
    septal_mask = binary_closing(septal_mask, iterations=1)
    
    return septal_mask

def analyze_ti_signals_for_mt(ti_list, signal_stack, mask=None, enhanced_detection=False):
    """
    Analyze TI signals for MT effects across an image with proper pixel-wise analysis
    
    Parameters:
    -----------
    ti_list : array
        List of inversion times
    signal_stack : array
        3D array of signals (n_TI, height, width)
    mask : array, optional
        ROI mask
    enhanced_detection : bool
        Whether to use enhanced detection methods
    
    Returns:
    --------
    mt_map : array
        Map of MT fractions
    confidence_map : array
        Map of detection confidence
    residual_stats : dict
        Statistics about residual analysis
    """
    import numpy as np
    from scipy.optimize import curve_fit
    
    # Get dimensions
    n_ti, height, width = signal_stack.shape
    
    # Create output maps
    mt_map = np.zeros((height, width))
    confidence_map = np.zeros((height, width))
    
    # Create mask if not provided
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    
    # Simple exponential model for fitting
    def exp_model(ti, a, b, t1):
        return a - b * np.exp(-ti / t1)
    
    # Count statistics
    total_pixels = 0
    mt_detected_count = 0
    correlations = []
    
    # Process each pixel
    print(f"Analyzing {np.sum(mask)} pixels for MT effects...")
    
    # Create correlation threshold based on method
    corr_threshold = 0.15 if enhanced_detection else 0.3
    
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
                
            # Get signal for this pixel
            pixel_signal = signal_stack[:, y, x]
            
            # Skip low-signal pixels
            if np.max(pixel_signal) < 1e-6:
                continue
                
            total_pixels += 1
            
            try:
                # Fit exponential model
                popt, _ = curve_fit(
                    exp_model, 
                    ti_list, 
                    pixel_signal,
                    p0=[np.max(pixel_signal), np.ptp(pixel_signal), 1000],
                    bounds=([0, 0, 100], [np.inf, np.inf, 3000]),
                    maxfev=100
                )
                
                # Calculate fitted values and residuals
                fitted = exp_model(ti_list, *popt)
                residuals = pixel_signal - fitted
                
                # Detect MT from residuals
                from scipy.stats import pearsonr
                
                # Look for systematic residual patterns
                if len(ti_list) > 3:
                    corr, p_val = pearsonr(ti_list[:-1], residuals[:-1])
                    mt_detected = abs(corr) > corr_threshold and p_val < 0.05
                    correlations.append(corr)
                    
                    # Only store confidence if correlation is high enough
                    confidence_map[y, x] = abs(corr) if abs(corr) > corr_threshold else 0
                    
                    if mt_detected:
                        # Use different MT estimation based on method
                        if enhanced_detection:
                            # Enhanced method - uses residual patterns
                            residual_amplitude = np.std(residuals)
                            signal_amplitude = np.mean(np.abs(residuals))
                            
                            # Calculate MT fraction
                            mt_frac = 0.1 * (residual_amplitude / max(signal_amplitude, 1e-6))
                            mt_frac = np.clip(mt_frac, 0.01, 0.15)  # More conservative clip
                        else:
                            # Standard method - uses simple empirical relationship
                            mt_frac = 0.05  # Default value
                            
                            # Adjust based on residual correlation
                            if abs(corr) > 0.5:
                                mt_frac = 0.1
                            elif abs(corr) > 0.7:
                                mt_frac = 0.15
                        
                        # Store MT fraction in map
                        mt_map[y, x] = mt_frac
                        mt_detected_count += 1
            except Exception as e:
                # Skip errors in individual pixels
                continue
    
    # Apply light smoothing to the MT map
    from scipy.ndimage import gaussian_filter
    mt_map_smoothed = gaussian_filter(mt_map, sigma=1.0)
    
    # Calculate statistics
    residual_stats = {
        'total_pixels_analyzed': total_pixels,
        'mt_detected_count': mt_detected_count,
        'detected_fraction': mt_detected_count / max(total_pixels, 1),
        'mean_correlation': np.mean(correlations) if correlations else 0,
        'std_correlation': np.std(correlations) if correlations else 0,
        'mean_correction_factor': 1.0 + np.mean(mt_map_smoothed) if np.any(mt_map_smoothed > 0) else 1.0
    }
    
    # Return smoothed maps
    return mt_map_smoothed, confidence_map, residual_stats

def plot_mt_effects(t1_map, t1_corrected, mt_map, confidence_map, 
                   roi_coords=None, save_path=None):
    """
    Visualize MT effects and correction
    
    Parameters:
    -----------
    t1_map : array
        Original T1 map
    t1_corrected : array
        Corrected T1 map
    mt_map : array
        MT correction factor map
    confidence_map : array
        Confidence map
    roi_coords : tuple, optional
        ROI coordinates for display
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original T1 map
    im1 = axes[0, 0].imshow(t1_map, cmap='jet', vmin=800, vmax=1400)
    axes[0, 0].set_title('Original T1 Map (ms)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Corrected T1 map
    im2 = axes[0, 1].imshow(t1_corrected, cmap='jet', vmin=800, vmax=1400)
    axes[0, 1].set_title('MT-Corrected T1 Map (ms)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference map
    diff_map = t1_corrected - t1_map
    im3 = axes[0, 2].imshow(diff_map, cmap='coolwarm', 
                           vmin=-100, vmax=100)
    axes[0, 2].set_title('T1 Difference (ms)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # MT correction factor map
    im4 = axes[1, 0].imshow(mt_map, cmap='hot', vmin=1.0, vmax=1.2)
    axes[1, 0].set_title('MT Correction Factor')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Confidence map
    im5 = axes[1, 1].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('Detection Confidence')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Histogram of corrections
    axes[1, 2].hist(diff_map[~np.isnan(diff_map)].flatten(), bins=50)
    axes[1, 2].set_xlabel('T1 Change (ms)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Distribution of T1 Changes')
    
    # Add ROI box if provided
    if roi_coords:
        x1, x2, y1, y2 = roi_coords
        for ax in axes.flat[:-1]:
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               edgecolor='white', facecolor='none',
                               linewidth=2, linestyle='--')
            ax.add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_residual_analysis(tis, signals, fitted, residuals, 
                         mt_detected=False, save_path=None):
    """
    Plot detailed residual analysis for MT detection
    
    Parameters:
    -----------
    tis : array
        Inversion times
    signals : array
        Measured signals
    fitted : array
        Fitted curve
    residuals : array
        Fitting residuals
    mt_detected : bool
        Whether MT was detected
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Signal and fit
    axes[0, 0].plot(tis, signals, 'bo', label='Data')
    axes[0, 0].plot(tis, fitted, 'r-', label='Fit')
    axes[0, 0].set_xlabel('TI (ms)')
    axes[0, 0].set_ylabel('Signal')
    axes[0, 0].set_title('Signal vs Fit')
    axes[0, 0].legend()
    
    # Residuals vs TI
    axes[0, 1].plot(tis, residuals, 'go-')
    axes[0, 1].axhline(y=0, color='k', linestyle='--')
    axes[0, 1].set_xlabel('TI (ms)')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'Residuals (MT {"Detected" if mt_detected else "Not Detected"})')
    
    # Residual distribution
    axes[1, 0].hist(residuals, bins=20)
    axes[1, 0].set_xlabel('Residual Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    
    # Autocorrelation of residuals
    autocorr = scipy_signal.correlate(residuals, residuals, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= autocorr[0]
    
    axes[1, 1].plot(autocorr[:min(len(autocorr), 10)])
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].set_title('Residual Autocorrelation')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def integrate_mt_correction(t1_map, signal_stack, TI_list, mask=None, 
                           apply_correction=True, plot_results=True):
    """
    Main integration function for MT correction
    
    Parameters:
    -----------
    t1_map : array
        Your existing T1 map
    signal_stack : array
        3D array of signals at each TI (n_TI, height, width)
    TI_list : array
        List of inversion times
    mask : array, optional
        Myocardium mask
    apply_correction : bool
        Whether to apply correction
    plot_results : bool
        Whether to plot results
    
    Returns:
    --------
    results : dict
        Dictionary containing all results
    """
    print("Analyzing MT effects...")
    
    # Analyze TI signals for MT
    mt_map, confidence_map, residual_stats = analyze_ti_signals_for_mt(
        TI_list, signal_stack, mask, enhanced_detection=False
    )
    
    print(f"MT detected in {residual_stats['detected_fraction']*100:.1f}% of pixels")
    print(f"Mean residual correlation: {residual_stats['mean_correlation']:.2f}")
    
    # Apply correction if requested
    if apply_correction:
        print("Applying MT correction...")
        t1_corrected = apply_mt_correction_to_t1map(t1_map, mt_map)
    else:
        # Uniform correction as fallback
        print("Applying uniform MT correction (10%)...")
        t1_corrected = apply_mt_correction_to_t1map(t1_map, uniform_correction=1.1)
    
    # Plot results if requested
    if plot_results:
        plot_mt_effects(t1_map, t1_corrected, mt_map, confidence_map)
    
    # Package results
    results = {
        't1_corrected': t1_corrected,
        'mt_map': mt_map,
        'confidence_map': confidence_map,
        'residual_stats': residual_stats,
        'correction_applied': apply_correction
    }
    
    return results
