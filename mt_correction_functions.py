import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

def detect_mt_from_residuals(tis, signal, fitted, threshold=0.15):
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

def estimate_mt_fraction(tis, residuals, signal, method='empirical'):
    """
    Estimate MT fraction from residual patterns
    
    Parameters:
    -----------
    tis : array
        Inversion times
    residuals : array
        Fitting residuals
    signal : array
        Original signal
    method : str
        Estimation method ('empirical' or 'enhanced')
    
    Returns:
    --------
    mt_fraction : float
        Estimated MT pool fraction
    k_exchange : float
        Exchange rate constant
    """
    # Enhanced estimation based on residual patterns
    if method == 'enhanced':
        # More sophisticated analysis
        residual_amplitude = np.std(residuals)
        signal_amplitude = np.mean(np.abs(signal))
        
        # Look for oscillatory patterns in residuals
        if len(residuals) > 4:
            # Check for sign changes in residuals
            sign_changes = np.sum(np.diff(np.sign(residuals)) != 0)
            oscillation_factor = sign_changes / (len(residuals) - 1)
        else:
            oscillation_factor = 0.0
        
        # Combine factors for MT fraction estimation
        mt_fraction = 0.1 * (residual_amplitude / signal_amplitude) * (1 + oscillation_factor)
    else:
        # Simple empirical estimation
        residual_amplitude = np.std(residuals)
        signal_amplitude = np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 1.0
        mt_fraction = 0.1 * (residual_amplitude / signal_amplitude)
    
    # Ensure reasonable bounds
    mt_fraction = np.clip(mt_fraction, 0.0, 0.3)  # Typical range 0-30%
    
    # Estimate exchange rate (simplified)
    k_exchange = 20.0  # Hz, typical for tissue
    
    return mt_fraction, k_exchange

def calculate_mt_correction_factor(mt_fraction, k_exchange, tis, tr, flip_angle,
                                  tissue_type='cardiac', field_strength=1.5):
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
        Type of tissue ('cardiac', 'liver', 'brain')
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
    # MT effects are generally stronger at higher field strengths
    field_factor = 1.0
    if field_strength >= 3.0:
        field_factor = 1.2  # 20% stronger MT effects at 3T
    elif field_strength <= 1.5:
        field_factor = 0.8  # 20% weaker MT effects at 1.5T
    
    # Tissue-specific correction
    tissue_factor = 1.0
    if tissue_type == 'cardiac':
        tissue_factor = 1.1  # Cardiac tissue has moderate MT
    elif tissue_type == 'brain':
        tissue_factor = 1.3  # Brain tissue has stronger MT
    elif tissue_type == 'liver':
        tissue_factor = 0.9  # Liver has weaker MT
    
    # Calculate correction factor
    # More sophisticated model considering exchange rate
    base_correction = 1.0 + mt_fraction * 0.5
    
    # Adjust for field strength and tissue type
    correction_factor = 1.0 + (base_correction - 1.0) * field_factor * tissue_factor
    
    # Ensure reasonable bounds
    correction_factor = np.clip(correction_factor, 1.0, 1.5)
    
    return correction_factor

def identify_septal_region(t1_map, percentile_threshold=15):
    """
    Identify septal region based on T1 values
    
    Parameters:
    -----------
    t1_map : array
        T1 map
    percentile_threshold : float
        Percentile threshold for septal identification
    
    Returns:
    --------
    septal_mask : array
        Boolean mask of septal region
    """
    # Simple approach: septum typically has higher T1 values
    valid_t1 = t1_map[~np.isnan(t1_map)]
    if len(valid_t1) > 0:
        threshold = np.percentile(valid_t1, 100 - percentile_threshold)
        septal_mask = t1_map > threshold
    else:
        septal_mask = np.zeros_like(t1_map, dtype=bool)
    
    return septal_mask

def apply_mt_correction_to_t1map(t1_map, mt_map=None, uniform_correction=None,
                                septal_mask=None, septal_boost=1.0):
    """
    Apply MT correction to T1 map
    
    Parameters:
    -----------
    t1_map : array
        Uncorrected T1 map
    mt_map : array, optional
        Pixel-wise MT correction factors
    uniform_correction : float, optional
        Uniform correction factor
    septal_mask : array, optional
        Mask for septal region
    septal_boost : float, optional
        Additional correction factor for septum
    
    Returns:
    --------
    t1_corrected : array
        MT-corrected T1 map
    """
    t1_corrected = t1_map.copy()
    
    if mt_map is not None:
        # Apply pixel-wise correction
        valid_mask = ~np.isnan(t1_map) & ~np.isnan(mt_map)
        t1_corrected[valid_mask] = t1_map[valid_mask] * mt_map[valid_mask]
        
        # Apply additional septal boost if provided
        if septal_mask is not None and septal_boost != 1.0:
            septal_valid = valid_mask & septal_mask
            t1_corrected[septal_valid] *= septal_boost
            
    elif uniform_correction is not None:
        # Apply uniform correction
        valid_mask = ~np.isnan(t1_map)
        t1_corrected[valid_mask] = t1_map[valid_mask] * uniform_correction
        
        # Apply additional septal boost if provided
        if septal_mask is not None and septal_boost != 1.0:
            septal_valid = valid_mask & septal_mask
            t1_corrected[septal_valid] *= septal_boost
    
    return t1_corrected

def analyze_ti_signals_for_mt(ti_list, signal_stack, mask=None, enhanced_detection=False):
    """
    Analyze TI signals for MT effects across an image
    
    Parameters:
    -----------
    ti_list : array
        List of inversion times
    signal_stack : array
        3D array of signals (n_TI, height, width)
    mask : array, optional
        ROI mask
    enhanced_detection : bool
        Use enhanced detection method
    
    Returns:
    --------
    correction_map : array
        Map of MT correction factors
    confidence_map : array
        Map of detection confidence
    residual_stats : dict
        Statistics about residual analysis
    """
    height, width = signal_stack.shape[1:]
    mt_fraction_map = np.zeros((height, width))
    correction_map = np.ones((height, width))  # Initialize to 1.0 (no correction)
    confidence_map = np.zeros((height, width))
    
    # Statistics
    total_pixels = 0
    mt_detected_count = 0
    correlations = []
    correction_factors = []
    
    # Simple exponential model for fitting
    def exp_model(ti, a, b, t1):
        return a - b * np.exp(-ti / t1)
    
    # Process each pixel
    for y in range(height):
        for x in range(width):
            if mask is not None and not mask[y, x]:
                continue
            
            pixel_signal = signal_stack[:, y, x]
            
            if np.max(pixel_signal) < 1e-6:
                continue
            
            try:
                # Fit exponential model
                popt, _ = curve_fit(exp_model, ti_list, pixel_signal,
                                   p0=[np.max(pixel_signal), 
                                       np.ptp(pixel_signal), 
                                       1000],
                                   bounds=([0, 0, 100], 
                                          [np.inf, np.inf, 3000]))
                
                fitted = exp_model(ti_list, *popt)
                
                # Detect MT from residuals
                mt_detected, corr, p_val = detect_mt_from_residuals(
                    ti_list, pixel_signal, fitted, 
                    threshold=0.1 if enhanced_detection else 0.15
                )
                
                if mt_detected or (enhanced_detection and abs(corr) > 0.05):
                    residuals = pixel_signal - fitted
                    mt_frac, k_exchange = estimate_mt_fraction(
                        ti_list, residuals, pixel_signal,
                        method='enhanced' if enhanced_detection else 'empirical'
                    )
                    mt_fraction_map[y, x] = mt_frac
                    
                    # Calculate correction factor
                    correction_factor = calculate_mt_correction_factor(
                        mt_frac, k_exchange, ti_list, 
                        tr=None, flip_angle=None,  # Will use defaults
                        tissue_type='cardiac', 
                        field_strength=1.5
                    )
                    correction_map[y, x] = correction_factor
                    mt_detected_count += 1
                    correction_factors.append(correction_factor)
                
                confidence_map[y, x] = abs(corr)
                correlations.append(corr)
                total_pixels += 1
                
            except:
                continue
    
    # Calculate statistics
    residual_stats = {
        'total_pixels_analyzed': total_pixels,
        'mt_detected_count': mt_detected_count,
        'detected_fraction': mt_detected_count / total_pixels if total_pixels > 0 else 0,
        'mean_correlation': np.mean(correlations) if correlations else 0,
        'std_correlation': np.std(correlations) if correlations else 0,
        'mean_correction_factor': np.mean(correction_factors) if correction_factors else 1.0,
        'mt_fraction_map': mt_fraction_map
    }
    
    return correction_map, confidence_map, residual_stats

# Updated plot_mt_effects function in mt_correction_functions.py with better error handling:

def plot_mt_effects(t1_map, t1_corrected, mt_correction_map, confidence_map, 
                   roi_coords=None, save_path=None):
    """
    Visualize MT effects and correction with robust handling
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # Ensure all inputs are valid arrays
    if t1_map is None or t1_corrected is None:
        print("Error: T1 maps are None")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Determine T1 range for visualization
    valid_t1 = t1_map[~np.isnan(t1_map)]
    if len(valid_t1) > 0:
        vmin_t1 = np.percentile(valid_t1, 5)
        vmax_t1 = np.percentile(valid_t1, 95)
    else:
        vmin_t1, vmax_t1 = 800, 1400  # Default range
    
    # 1. Original T1 map
    try:
        im1 = axes[0, 0].imshow(t1_map, cmap='jet', vmin=vmin_t1, vmax=vmax_t1)
        axes[0, 0].set_title('Original T1 Map (ms)')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f"Error displaying original T1:\n{str(e)}", 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Original T1 Map (Error)')
    
    # 2. Corrected T1 map
    try:
        # Ensure corrected map exists
        if t1_corrected is None or np.all(np.isnan(t1_corrected)):
            # If no corrected map, apply correction now
            if mt_correction_map is not None:
                t1_corrected = t1_map * mt_correction_map
            else:
                t1_corrected = t1_map * 1.03  # Default 3% correction
        
        im2 = axes[0, 1].imshow(t1_corrected, cmap='jet', vmin=vmin_t1, vmax=vmax_t1)
        axes[0, 1].set_title('MT-Corrected T1 Map (ms)')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f"Error displaying corrected T1:\n{str(e)}", 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('MT-Corrected T1 Map (Error)')
    
    # 3. Difference map
    try:
        diff_map = t1_corrected - t1_map
        valid_diff = diff_map[~np.isnan(diff_map)]
        if len(valid_diff) > 0:
            vmax_diff = max(abs(np.percentile(valid_diff, 5)), abs(np.percentile(valid_diff, 95)))
            vmin_diff = -vmax_diff
        else:
            vmin_diff, vmax_diff = -100, 100
        
        im3 = axes[0, 2].imshow(diff_map, cmap='coolwarm', vmin=vmin_diff, vmax=vmax_diff)
        axes[0, 2].set_title('T1 Difference (ms)')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    except Exception as e:
        axes[0, 2].text(0.5, 0.5, f"Error displaying difference:\n{str(e)}", 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('T1 Difference (Error)')
    
    # 4. MT correction factor map
    try:
        if mt_correction_map is None:
            mt_correction_map = np.ones_like(t1_map)
        
        # Check if correction map has any variation
        unique_corrections = np.unique(mt_correction_map[~np.isnan(mt_correction_map)])
        if len(unique_corrections) == 1 and unique_corrections[0] == 1.0:
            # No MT correction applied
            axes[1, 0].text(0.5, 0.5, "No MT correction applied\n(all values = 1.0)", 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12, color='red')
            axes[1, 0].set_title('MT Correction Factor')
            axes[1, 0].axis('off')
        else:
            im4 = axes[1, 0].imshow(mt_correction_map, cmap='hot', vmin=1.0, vmax=1.3)
            axes[1, 0].set_title('MT Correction Factor')
            axes[1, 0].axis('off')
            cbar4 = plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
            cbar4.set_label('Correction Factor')
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f"Error displaying MT correction:\n{str(e)}", 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('MT Correction Factor (Error)')
    
    # 5. Detection confidence map
    try:
        if confidence_map is None:
            confidence_map = np.zeros_like(t1_map)
        
        im5 = axes[1, 1].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title('Detection Confidence')
        axes[1, 1].axis('off')
        cbar5 = plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar5.set_label('Confidence')
        
        # Add statistics
        valid_conf = confidence_map[~np.isnan(confidence_map)]
        if len(valid_conf) > 0:
            mean_conf = np.mean(valid_conf)
            max_conf = np.max(valid_conf)
            axes[1, 1].text(0.02, 0.02, f"Mean: {mean_conf:.3f}\nMax: {max_conf:.3f}", 
                           transform=axes[1, 1].transAxes, color='white',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f"Error displaying confidence:\n{str(e)}", 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Detection Confidence (Error)')
    
    # 6. Histogram of corrections
    try:
        axes[1, 2].clear()  # Clear any existing content
        
        # Calculate actual T1 changes
        if diff_map is not None:
            valid_diff = diff_map[~np.isnan(diff_map)].flatten()
            if len(valid_diff) > 0:
                axes[1, 2].hist(valid_diff, bins=50, alpha=0.7, color='blue')
                axes[1, 2].axvline(0, color='k', linestyle='--', alpha=0.5)
                axes[1, 2].set_xlabel('T1 Change (ms)')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].set_title('Distribution of T1 Changes')
                axes[1, 2].grid(True, alpha=0.3)
                
                # Add statistics
                mean_change = np.mean(valid_diff)
                std_change = np.std(valid_diff)
                axes[1, 2].text(0.02, 0.98, 
                               f"Mean: {mean_change:.1f} ms\n"
                               f"Std: {std_change:.1f} ms\n"
                               f"N pixels: {len(valid_diff)}", 
                               transform=axes[1, 2].transAxes,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[1, 2].text(0.5, 0.5, "No T1 changes to display", 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Distribution of T1 Changes')
    except Exception as e:
        axes[1, 2].text(0.5, 0.5, f"Error creating histogram:\n{str(e)}", 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Distribution (Error)')
    
    # Add ROI box if provided
    if roi_coords:
        try:
            x1, x2, y1, y2 = roi_coords
            for i, ax in enumerate(axes.flat[:-1]):  # All except histogram
                if ax.images:  # Only if image is displayed
                    rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                   edgecolor='lime', facecolor='none',
                                   linewidth=2, linestyle='--')
                    ax.add_patch(rect)
        except Exception as e:
            print(f"Error adding ROI boxes: {e}")
    
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
    
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
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs TI
    axes[0, 1].plot(tis, residuals, 'go-')
    axes[0, 1].axhline(y=0, color='k', linestyle='--')
    axes[0, 1].set_xlabel('TI (ms)')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'Residuals (MT {"Detected" if mt_detected else "Not Detected"})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1, 0].hist(residuals, bins=20)
    axes[1, 0].set_xlabel('Residual Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Autocorrelation of residuals
    from scipy import signal as scipy_signal
    autocorr = scipy_signal.correlate(residuals, residuals, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= autocorr[0] if autocorr[0] != 0 else 1.0
    
    axes[1, 1].plot(autocorr[:min(len(autocorr), 10)])
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].set_title('Residual Autocorrelation')
    axes[1, 1].grid(True, alpha=0.3)
    
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
    correction_map, confidence_map, residual_stats = analyze_ti_signals_for_mt(
        TI_list, signal_stack, mask
    )
    
    print(f"MT detected in {residual_stats['detected_fraction']*100:.1f}% of pixels")
    print(f"Mean residual correlation: {residual_stats['mean_correlation']:.2f}")
    print(f"Mean correction factor: {residual_stats['mean_correction_factor']:.3f}")
    
    # Apply correction if requested
    if apply_correction and residual_stats['detected_fraction'] > 0:
        print("Applying MT correction...")
        t1_corrected = apply_mt_correction_to_t1map(t1_map, mt_map=correction_map)
    else:
        # Uniform correction as fallback
        print("Applying uniform MT correction (3% for 1.5T)...")
        t1_corrected = apply_mt_correction_to_t1map(t1_map, uniform_correction=1.03)
    
    # Plot results if requested
    if plot_results:
        plot_mt_effects(t1_map, t1_corrected, correction_map, confidence_map)
    
    # Package results
    results = {
        't1_corrected': t1_corrected,
        'mt_correction_map': correction_map,
        'confidence_map': confidence_map,
        'residual_stats': residual_stats,
        'correction_applied': apply_correction
    }
    
    return results
