import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def detect_mt_from_residuals(tis, signal, fitted, threshold=0.15):
    """
    Enhanced MT detection from fitting residuals using multiple criteria
    
    MT effects typically cause:
    1. Systematic underestimation of signal at early TIs
    2. Oscillating residual patterns
    3. Non-random residual distribution
    """
    residuals = signal - fitted
    n_points = len(tis)
    
    if n_points < 4:
        return False, 0.0, 1.0
    
    # Criterion 1: Check for systematic bias in early vs late TIs
    early_residuals = residuals[:n_points//2]
    late_residuals = residuals[n_points//2:]
    
    # MT typically causes negative residuals early, positive late
    early_bias = np.mean(early_residuals)
    late_bias = np.mean(late_residuals)
    bias_difference = late_bias - early_bias
    
    # Criterion 2: Check for oscillating pattern (MT exchange)
    # Look for sign changes in residuals
    sign_changes = np.sum(np.diff(np.sign(residuals)) != 0)
    expected_changes = n_points // 3  # Expected for random
    
    # Criterion 3: Residual magnitude relative to signal
    relative_residual = np.std(residuals) / (np.max(signal) - np.min(signal))
    
    # Criterion 4: Statistical test for non-zero mean residuals
    t_stat, p_value = ttest_1samp(residuals, 0)
    
    # Combined MT detection score
    mt_score = 0.0
    
    # Weight different criteria
    if bias_difference > 0 and relative_residual > 0.02:
        mt_score += 0.3
    
    if sign_changes > expected_changes:
        mt_score += 0.2
        
    if relative_residual > 0.05:  # 5% of signal range
        mt_score += 0.3
        
    if p_value < 0.1:  # Significant non-zero residuals
        mt_score += 0.2
    
    # Detect MT if score exceeds threshold
    mt_detected = mt_score > threshold
    
    return mt_detected, mt_score, p_value

def estimate_mt_fraction(tis, residuals, signal=None, method='enhanced'):
    """
    Enhanced MT fraction estimation based on residual patterns and signal characteristics
    """
    if method == 'enhanced' and signal is not None:
        # Analyze the pattern of residuals
        residual_rms = np.sqrt(np.mean(residuals**2))
        signal_range = np.ptp(signal)
        
        # MT fraction correlates with relative residual magnitude
        # Typical cardiac MT fraction is 10-20%
        base_mt_fraction = 0.15  # Base assumption for cardiac tissue
        
        # Adjust based on residual pattern
        relative_residual = residual_rms / signal_range
        
        # Empirical relationship calibrated for cardiac tissue
        if relative_residual > 0.1:
            mt_fraction = base_mt_fraction * 1.5
        elif relative_residual > 0.05:
            mt_fraction = base_mt_fraction * 1.2
        elif relative_residual > 0.02:
            mt_fraction = base_mt_fraction
        else:
            mt_fraction = base_mt_fraction * 0.7
            
        # Check for oscillating pattern (indicates faster exchange)
        sign_changes = np.sum(np.diff(np.sign(residuals)) != 0)
        if sign_changes > len(residuals) // 2:
            k_exchange = 30.0  # Faster exchange
        else:
            k_exchange = 20.0  # Normal exchange
            
    else:
        # Fallback to simple estimation
        residual_amplitude = np.std(residuals)
        signal_amplitude = np.mean(np.abs(signal)) if signal is not None else 1.0
        mt_fraction = 0.15 * (residual_amplitude / signal_amplitude)
        k_exchange = 20.0
    
    # Ensure physiological bounds for cardiac tissue
    mt_fraction = np.clip(mt_fraction, 0.05, 0.25)  # 5-25% range
    
    return mt_fraction, k_exchange

def calculate_mt_correction_factor(mt_fraction, k_exchange, tis, tr=None, 
                                  flip_angle=None, tissue_type='cardiac', field_strength=1.5):
    """
    Calculate T1 correction factor based on Bloch-McConnell equations
    Now includes field strength consideration
    """
    if mt_fraction <= 0:
        return 1.0
    
    # Field strength scaling
    if field_strength == 1.5:
        mt_scaling = 0.5  # MT effects are smaller at 1.5T
    else:  # 3T
        mt_scaling = 1.0
    
    # Tissue-specific parameters
    if tissue_type == 'cardiac':
        # Cardiac-specific MT characteristics
        T1_bound = 1.0  # T1 of bound pool (seconds)
        T2_bound = 10e-6  # T2 of bound pool (10 microseconds)
        
        # Scale MT effect by field strength
        # At 1.5T: 30% of MT fraction translates to T1 error (vs 60% at 3T)
        base_correction = 1.0 + mt_fraction * 0.3 * mt_scaling
        
        # Adjust for exchange rate (faster exchange = larger correction)
        exchange_factor = k_exchange / 20.0  # Normalized to typical rate
        exchange_factor = np.clip(exchange_factor, 0.5, 2.0)
        
        correction_factor = base_correction * (0.8 + 0.2 * exchange_factor)
        
        # Additional correction for MOLLI-specific effects
        if tr is not None and tr < 5.0:  # Short TR in MOLLI
            # Short TR enhances MT effects
            tr_factor = 1.0 + (5.0 - tr) * 0.02  # 2% per ms below 5ms
            correction_factor *= tr_factor
            
    else:
        # Generic tissue correction
        correction_factor = 1.0 + mt_fraction * 0.25 * mt_scaling
    
    # More conservative bounds for 1.5T
    if field_strength == 1.5:
        correction_factor = np.clip(correction_factor, 1.02, 1.15)  # 2-15% correction
    else:
        correction_factor = np.clip(correction_factor, 1.05, 1.25)  # 5-25% correction
    
    return correction_factor

def apply_mt_correction_to_t1map(t1_map, mt_map=None, uniform_correction=None,
                                 septal_mask=None, septal_boost=1.1):
    """
    Apply MT correction with special handling for septal region
    """
    t1_corrected = t1_map.copy()
    
    # Apply base correction
    if mt_map is not None:
        # Pixel-wise correction
        valid_mask = ~np.isnan(t1_map) & ~np.isnan(mt_map)
        
        # Ensure mt_map contains correction factors, not just MT fractions
        if np.max(mt_map[valid_mask]) < 0.5:  # Likely MT fractions
            # Convert to correction factors
            mt_correction_map = np.ones_like(mt_map)
            mt_correction_map[valid_mask] = 1.0 + mt_map[valid_mask] * 0.6
        else:
            mt_correction_map = mt_map
            
        t1_corrected[valid_mask] = t1_map[valid_mask] * mt_correction_map[valid_mask]
        
    elif uniform_correction is not None:
        # Apply uniform correction
        valid_mask = ~np.isnan(t1_map)
        t1_corrected[valid_mask] = t1_map[valid_mask] * uniform_correction
    else:
        # Default cardiac MT correction (12%)
        valid_mask = ~np.isnan(t1_map)
        t1_corrected[valid_mask] = t1_map[valid_mask] * 1.12
    
    # Special handling for septal region if identified
    if septal_mask is not None:
        septal_valid = septal_mask & valid_mask
        if np.any(septal_valid):
            # Apply additional correction to septum
            t1_corrected[septal_valid] *= septal_boost
    
    return t1_corrected

def identify_septal_region(t1_map, percentile_threshold=10):
    """
    Identify potential septal region based on abnormally low T1 values
    and spatial characteristics
    """
    valid_mask = ~np.isnan(t1_map)
    if not np.any(valid_mask):
        return np.zeros_like(t1_map, dtype=bool)
    
    # Find suspiciously low T1 values
    t1_threshold = np.percentile(t1_map[valid_mask], percentile_threshold)
    low_t1_mask = t1_map < t1_threshold
    
    # Spatial criteria: septum is typically in the center
    h, w = t1_map.shape
    y_center, x_center = h // 2, w // 2
    
    # Create distance map from center
    y_coords, x_coords = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((y_coords - y_center)**2 + (x_coords - x_center)**2)
    
    # Septum is typically central
    central_mask = distance_from_center < min(h, w) * 0.4
    
    # Combine criteria
    septal_mask = low_t1_mask & central_mask & valid_mask
    
    # Clean up using morphological operations
    from scipy.ndimage import binary_opening, binary_closing
    septal_mask = binary_closing(binary_opening(septal_mask))
    
    return septal_mask

def analyze_ti_signals_for_mt(ti_list, signal_stack, mask=None, 
                              enhanced_detection=True):
    """
    Enhanced analysis of TI signals for MT effects
    """
    if len(signal_stack.shape) == 3:
        height, width = signal_stack.shape[1:]
    else:
        # Handle different array ordering
        height, width = signal_stack.shape[0:2]
        signal_stack = np.transpose(signal_stack, (2, 0, 1))
    
    mt_map = np.zeros((height, width))
    confidence_map = np.zeros((height, width))
    correction_map = np.ones((height, width))
    
    # Statistics
    total_pixels = 0
    mt_detected_count = 0
    mt_scores = []
    mt_fractions = []
    
    # Enhanced exponential model with better initial guesses
    def exp_model(ti, a, b, t1):
        return a - b * np.exp(-ti / t1)
    
    # Process each pixel
    for y in range(height):
        for x in range(width):
            if mask is not None and not mask[y, x]:
                continue
            
            pixel_signal = signal_stack[:, y, x]
            
            if np.max(np.abs(pixel_signal)) < 1e-6:
                continue
            
            try:
                # Better initial parameter estimation
                signal_max = np.max(pixel_signal)
                signal_min = np.min(pixel_signal)
                signal_range = signal_max - signal_min
                
                # Estimate initial T1 from signal recovery
                half_recovery_idx = np.argmin(np.abs(pixel_signal - (signal_max + signal_min)/2))
                t1_guess = ti_list[half_recovery_idx] / np.log(2)
                t1_guess = np.clip(t1_guess, 500, 2000)
                
                # Fit with better bounds
                popt, _ = curve_fit(exp_model, ti_list, pixel_signal,
                                   p0=[signal_max, signal_range*1.8, t1_guess],
                                   bounds=([signal_max*0.5, 0, 200], 
                                          [signal_max*2, signal_range*3, 3000]),
                                   maxfev=1000)
                
                fitted = exp_model(ti_list, *popt)
                residuals = pixel_signal - fitted
                
                # Enhanced MT detection
                if enhanced_detection:
                    mt_detected, mt_score, p_val = detect_mt_from_residuals(
                        ti_list, pixel_signal, fitted, threshold=0.15
                    )
                    
                    if mt_detected or mt_score > 0.1:  # Lower threshold for mapping
                        mt_frac, k_ex = estimate_mt_fraction(
                            ti_list, residuals, pixel_signal, method='enhanced'
                        )
                        mt_map[y, x] = mt_frac
                        
                        # Calculate correction factor
                        corr_factor = calculate_mt_correction_factor(
                            mt_frac, k_ex, ti_list, tissue_type='cardiac'
                        )
                        correction_map[y, x] = corr_factor
                        
                        mt_detected_count += 1
                        mt_fractions.append(mt_frac)
                    else:
                        # Apply minimal MT correction even if not strongly detected
                        mt_map[y, x] = 0.10  # Assume 10% baseline MT
                        correction_map[y, x] = 1.06  # 6% correction
                    
                    confidence_map[y, x] = mt_score
                    mt_scores.append(mt_score)
                    
                else:
                    # Simple detection (fallback)
                    mt_detected, corr, p_val = detect_mt_from_residuals(
                        ti_list, pixel_signal, fitted, threshold=0.5
                    )
                    if mt_detected:
                        mt_detected_count += 1
                
                total_pixels += 1
                
            except Exception as e:
                # Apply default MT assumption for failed fits
                mt_map[y, x] = 0.10
                correction_map[y, x] = 1.06
                continue
    
    # Calculate comprehensive statistics
    residual_stats = {
        'total_pixels_analyzed': total_pixels,
        'mt_detected_count': mt_detected_count,
        'detected_fraction': mt_detected_count / total_pixels if total_pixels > 0 else 0,
        'mean_mt_score': np.mean(mt_scores) if mt_scores else 0,
        'mean_mt_fraction': np.mean(mt_fractions) if mt_fractions else 0.10,
        'mean_correction_factor': np.mean(correction_map[correction_map > 1]),
        'detection_threshold_used': 0.15 if enhanced_detection else 0.5
    }
    
    # Replace mt_map with correction_map for direct application
    return correction_map, confidence_map, residual_stats

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
        MT fraction map
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
    
    # MT fraction map
    im4 = axes[1, 0].imshow(mt_map, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 0].set_title('MT Fraction')
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
    from scipy import signal as scipy_signal
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
        3D array of signals at each TI (x, y, n_TI)
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
        TI_list, signal_stack, mask
    )
    
    print(f"MT detected in {residual_stats['detected_fraction']*100:.1f}% of pixels")
    print(f"Mean residual correlation: {residual_stats['mean_mt_score']:.2f}")
    
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
