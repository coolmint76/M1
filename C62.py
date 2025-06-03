import tkinter as tk
import SimpleITK as sitk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import datetime
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import traceback
import warnings
import datetime
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Button
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter
import scipy.interpolate
import cv2
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
# Add MT correction functions import
from mt_correction_functions import (
    detect_mt_from_residuals,
    estimate_mt_fraction,
    calculate_mt_correction_factor,
    apply_mt_correction_to_t1map,
    analyze_ti_signals_for_mt,
    plot_mt_effects,
    plot_residual_analysis,
    integrate_mt_correction,
    identify_septal_region
)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def load_dual_inversion_angle_images(fa1_paths, fa2_paths):
    """
    Load two sets of DICOM images with different inversion flip angles
    
    Parameters:
    -----------
    fa1_paths : list
        Paths to DICOM files for first inversion flip angle
    fa2_paths : list
        Paths to DICOM files for second inversion flip angle
            
    Returns:
    --------
    image_fa1 : ndarray
        Average image from first inversion flip angle series
    image_fa2 : ndarray
        Average image from second inversion flip angle series
    fa1 : float
        First inversion flip angle in degrees
    fa2 : float
        Second inversion flip angle in degrees
    inversion_times : list
        List of inversion times for both series
    """
    import numpy as np
    import pydicom
    
    # Load first flip angle series
    images_fa1 = []
    inversion_angles_1 = []
    inversion_times_1 = []
    
    for path in fa1_paths:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        
        # Apply rescale if available
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        img = img * slope + intercept
        
        images_fa1.append(img)
        
        # Try to get inversion angles - might be in private tags or elsewhere
        # For now, assume we know it's 30 degrees
        inversion_angles_1.append(30.0)
        
        # Get inversion time
        if hasattr(ds, 'InversionTime'):
            inversion_times_1.append(float(ds.InversionTime))
    
    # Load second flip angle series
    images_fa2 = []
    inversion_angles_2 = []
    inversion_times_2 = []
    
    for path in fa2_paths:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        
        # Apply rescale if available
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        img = img * slope + intercept
        
        images_fa2.append(img)
        
        # Try to get inversion angles - might be in private tags or elsewhere
        # For now, assume we know it's 60 degrees
        inversion_angles_2.append(60.0)
        
        # Get inversion time
        if hasattr(ds, 'InversionTime'):
            inversion_times_2.append(float(ds.InversionTime))
    
    # Check dimensions match
    shapes_1 = [img.shape for img in images_fa1]
    shapes_2 = [img.shape for img in images_fa2]
    
    if len(set(shapes_1)) > 1 or len(set(shapes_2)) > 1:
        raise ValueError("All images in each series must have the same dimensions")
    
    if shapes_1[0] != shapes_2[0]:
        raise ValueError("Images from both flip angle series must have the same dimensions")
    
    # Average images if multiple slices
    if len(images_fa1) > 1:
        image_fa1 = np.mean(np.stack(images_fa1, axis=0), axis=0)
    else:
        image_fa1 = images_fa1[0]
    
    if len(images_fa2) > 1:
        image_fa2 = np.mean(np.stack(images_fa2, axis=0), axis=0)
    else:
        image_fa2 = images_fa2[0]
    
    # Get inversion flip angles
    if inversion_angles_1:
        fa1 = np.mean(inversion_angles_1)
    else:
        # Default to 30 degrees if not found
        fa1 = 30.0
    
    if inversion_angles_2:
        fa2 = np.mean(inversion_angles_2)
    else:
        # Default to 60 degrees if not found
        fa2 = 60.0
    
    # Ensure flip angles are different
    if abs(fa1 - fa2) < 5:
        raise ValueError(f"Inversion flip angles too similar: {fa1}° and {fa2}°. Need at least 5° difference.")
    
    # Combine inversion times if available
    inversion_times = []
    if inversion_times_1:
        inversion_times.extend(inversion_times_1)
    if inversion_times_2:
        inversion_times.extend(inversion_times_2)
    
    return image_fa1, image_fa2, fa1, fa2, inversion_times

def calculate_b1_map_from_inversion_angles(image_fa1, image_fa2, fa1_deg, fa2_deg, inversion_times=None, mask_threshold=0.05):
    """
    Calculate B1 map from two images with different inversion pulse flip angles
    with much more robust error handling
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    print(f"Starting B1 calculation with angles {fa1_deg}° and {fa2_deg}°")
    
    # Ensure images have the same shape
    if image_fa1.shape != image_fa2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Use more robust preprocessing
    # Replace any negative values or zeros with small positive values
    image_fa1_clean = np.maximum(image_fa1, 1e-6)
    image_fa2_clean = np.maximum(image_fa2, 1e-6)
    
    # Scale images to similar intensity levels if needed
    if np.max(image_fa1_clean) / np.max(image_fa2_clean) > 10 or np.max(image_fa2_clean) / np.max(image_fa1_clean) > 10:
        print("Large intensity difference between images - normalizing")
        image_fa1_clean = image_fa1_clean / np.max(image_fa1_clean)
        image_fa2_clean = image_fa2_clean / np.max(image_fa2_clean)
    
    # Create mask based on signal intensity - use very permissive threshold
    max_signal1 = np.max(image_fa1_clean)
    max_signal2 = np.max(image_fa2_clean)
    signal_mask = ((image_fa1_clean > mask_threshold * max_signal1) & 
                   (image_fa2_clean > mask_threshold * max_signal2))
    
    # If mask is too restrictive, relax it
    if np.sum(signal_mask) < 100:  # Too few pixels
        print(f"Mask too restrictive with {np.sum(signal_mask)} pixels - relaxing threshold")
        mask_threshold /= 5  # Try with much lower threshold
        signal_mask = ((image_fa1_clean > mask_threshold * max_signal1) & 
                       (image_fa2_clean > mask_threshold * max_signal2))
        print(f"New mask has {np.sum(signal_mask)} pixels")
    
    # Initialize B1 map and confidence map
    b1_map = np.ones_like(image_fa1, dtype=np.float32)
    confidence_map = np.zeros_like(image_fa1, dtype=np.float32)
    
    # Convert flip angles to radians
    fa1_rad = np.radians(fa1_deg)
    fa2_rad = np.radians(fa2_deg)
    
    # Calculate sin values for reference angles
    sin_fa1 = np.sin(fa1_rad)
    sin_fa2 = np.sin(fa2_rad)
    
    if sin_fa1 < 1e-6 or sin_fa2 < 1e-6:
        print("Warning: One of the flip angles is too small for reliable calculation")
        # Use default values
        return b1_map, confidence_map
    
    # Calculate ratio of signals within the mask
    ratio = np.ones_like(image_fa1, dtype=np.float32)
    
    # Calculate ratio only for non-zero values in the denominator
    valid_division = signal_mask & (image_fa2_clean > 1e-6)
    if np.any(valid_division):
        ratio[valid_division] = image_fa1_clean[valid_division] / image_fa2_clean[valid_division]
    
    # Apply ratio threshold to avoid extreme values
    valid_ratio = (ratio > 0.1) & (ratio < 10) & signal_mask
    
    # Only proceed if we have some valid pixels
    if np.sum(valid_ratio) < 10:
        print(f"Too few valid ratio pixels: {np.sum(valid_ratio)}. Using default B1 map.")
        # Return a uniform B1 map
        return b1_map, confidence_map
    
    print(f"Found {np.sum(valid_ratio)} valid pixels for B1 calculation")
    
    # Calculate B1 factor for each valid pixel with robust error handling
    try:
        # Calculate the argument for arcsin: ratio * sin_fa2 / sin_fa1
        arcsin_arg = np.zeros_like(ratio)
        arcsin_arg[valid_ratio] = ratio[valid_ratio] * sin_fa2 / sin_fa1
        
        # Clip to valid arcsin range with a safety margin
        arcsin_arg_clipped = np.clip(arcsin_arg, -0.95, 0.95)
        
        # Apply the B1 calculation formula
        b1_map[valid_ratio] = (2.0 / np.pi) * np.arcsin(arcsin_arg_clipped[valid_ratio]) * (np.pi / 2.0)
        
        # Create confidence map based on signal strength and ratio validity
        confidence_term = 1.0 - np.abs(arcsin_arg_clipped - 0.5) / 0.5
        confidence_map[valid_ratio] = np.clip(confidence_term[valid_ratio], 0.0, 1.0)
        
        # Apply smoothing to reduce noise
        b1_smoothed = gaussian_filter(b1_map, sigma=1.0)
        
        # Ensure B1 values are in reasonable range
        b1_smoothed = np.clip(b1_smoothed, 0.5, 1.5)
        
        # Adjust confidence based on signal strength
        avg_signal = (image_fa1_clean + image_fa2_clean) / 2
        signal_confidence = np.clip(avg_signal / (0.7 * max(max_signal1, max_signal2)), 0, 1)
        confidence_map = confidence_map * signal_confidence
        
        return b1_smoothed, confidence_map
        
    except Exception as e:
        print(f"Error in B1 calculation: {e}")
        # If calculation fails, return a default B1 map
        return b1_map, confidence_map


def ir_model_simple(ti, a, b, t1, efficiency=1.0):
    """Simple inversion recovery model with optional efficiency parameter
    
    Parameters:
    -----------
    ti : float or ndarray
        Inversion time in ms
    a : float
        Signal amplitude parameter
    b : float
        Inversion factor
    t1 : float
        T1 relaxation time in ms
    efficiency : float, optional
        Inversion efficiency (0-1), default=1.0
        
    Returns:
    --------
    signal : float or ndarray
        Predicted signal intensity
    """
    return a - b * efficiency * np.exp(-ti / t1)

def ir_model_classic(ti, a, b, t1, efficiency=1.0):
    """Classic inversion recovery model with optional efficiency parameter
    
    Parameters:
    -----------
    ti : float or ndarray
        Inversion time in ms
    a : float
        Signal amplitude parameter
    b : float
        Inversion factor (0-1)
    t1 : float
        T1 relaxation time in ms
    efficiency : float, optional
        Inversion efficiency (0-1), default=1.0
        
    Returns:
    --------
    signal : float or ndarray
        Predicted signal intensity
    """
    return a * (1 - 2 * b * efficiency * np.exp(-ti / t1))

# IR model function
def ir_model(ti, a, b, t1, efficiency):
    """Inversion recovery model function"""
    adjusted_b = b * efficiency
    return a - adjusted_b * np.exp(-ti / t1)

def show_error_dialog(title, message, exception=None):
    """Shows a detailed error dialog with option to view traceback"""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    error_dialog = tk.Toplevel(root)
    error_dialog.title(title)
    error_dialog.geometry("500x400")
    
    # Main message
    ttk.Label(error_dialog, text=message, font=("Arial", 11), 
             wraplength=450).pack(padx=20, pady=10)
    
    # Create a frame for the details
    details_frame = ttk.LabelFrame(error_dialog, text="Error Details")
    
    if exception:
        # Add traceback text widget
        traceback_text = tk.Text(details_frame, height=10, width=60, wrap=tk.WORD)
        traceback_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Insert error details
        error_details = f"Error: {str(exception)}\n\n"
        error_details += traceback.format_exc()
        traceback_text.insert(tk.END, error_details)
        traceback_text.config(state=tk.DISABLED)  # Make read-only
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(traceback_text, command=traceback_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        traceback_text.config(yscrollcommand=scrollbar.set)
        
        # Show details frame
        details_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
    
    # Add close button
    ttk.Button(error_dialog, text="Close", 
              command=lambda: [error_dialog.destroy(), root.destroy()]).pack(pady=15)
    
    # Make sure this window stays on top
    error_dialog.transient(root)
    try:
        error_dialog.grab_set()
    except Exception:
        # Silently ignore grab failures
        pass
    
    root.wait_window(error_dialog)

def calculate_look_locker_correction_factor(t1_apparent, flip_angle, tr, model_type='MOLLI'):
    """
    Calculate Look-Locker correction factor for GRE-based MOLLI sequences
    Based on Messroghli et al. MRM 2004 and subsequent MOLLI literature
    
    Parameters:
    -----------
    t1_apparent : float or array
        Apparent T1 from exponential fit (ms)
    flip_angle : float
        Readout flip angle in degrees
    tr : float
        Repetition time in ms
    model_type : str
        'MOLLI' for standard correction, 'MOLLI_3param' for 3-parameter fit
    
    Returns:
    --------
    correction_factor : float or array
        Multiplicative correction factor to get true T1
    """
    import numpy as np
    
    if flip_angle is None or tr is None:
        # Default correction based on typical MOLLI parameters
        return 1.4  # Empirical value from literature
    
    # Convert flip angle to radians
    alpha_rad = np.radians(flip_angle)
    
    if model_type == 'MOLLI':
        # Standard MOLLI correction (Messroghli et al.)
        # T1* = T1 * (B/A - 1) where A and B are from 3-parameter fit
        # For 2-parameter fit: T1 = T1* * (1 - cos(alpha) * exp(-TR/T1))
        
        cos_alpha = np.cos(alpha_rad)
        
        # Iterative solution for true T1
        # Start with apparent T1
        t1_true = t1_apparent
        
        # Iterate to solve: T1* = T1 * (1 - cos(alpha) * exp(-TR/T1))
        for _ in range(5):  # Usually converges in 3-4 iterations
            exp_term = np.exp(-tr / t1_true)
            t1_true = t1_apparent / (1 - cos_alpha * exp_term)
            
            # Ensure reasonable bounds
            if isinstance(t1_true, np.ndarray):
                t1_true = np.clip(t1_true, t1_apparent, t1_apparent * 2.0)
            else:
                t1_true = max(t1_apparent, min(t1_true, t1_apparent * 2.0))
        
        correction_factor = t1_true / t1_apparent
        
    elif model_type == 'MOLLI_3param':
        # For 3-parameter fit: A(1 - B*exp(-t/T1*))
        # Correction factor = B/(B-1) for perfect inversion
        # This requires the B parameter from the fit
        # Default to standard correction if B not available
        correction_factor = 1.4
    
    else:
        # Default empirical correction
        correction_factor = 1.4
    
    # Ensure correction factor is within reasonable bounds
    # Based on literature, LL correction for MOLLI is typically 1.1-1.5
    if isinstance(correction_factor, np.ndarray):
        correction_factor = np.clip(correction_factor, 1.1, 1.5)
    else:
        correction_factor = max(1.1, min(1.5, correction_factor))
    
    return correction_factor

def estimate_t2star_bias_in_molli(te, t2star, readout_duration=None, n_phase_encodes=None):
    """
    Estimate T2* bias in MOLLI T1 measurements
    T2* doesn't change T1, but causes signal decay that can bias T1 estimation
    
    Parameters:
    -----------
    te : float
        Echo time in ms
    t2star : float
        T2* value in ms
    readout_duration : float, optional
        Total readout duration in ms (for multi-echo effects)
    n_phase_encodes : int, optional
        Number of phase encoding steps
    
    Returns:
    --------
    bias_correction : float
        Multiplicative correction factor for T1 bias due to T2* decay
    """
    import numpy as np
    
    if te is None or t2star is None or t2star <= 0:
        return 1.0
    
    # Basic T2* decay at echo time
    signal_loss_at_te = np.exp(-te / t2star)
    
    # For GRE readout, signal decays during readout
    if readout_duration is not None and readout_duration > 0:
        # Average signal loss over readout duration
        # Approximate as exponential decay from TE to TE + readout_duration
        te_end = te + readout_duration
        signal_loss_end = np.exp(-te_end / t2star)
        avg_signal_loss = (signal_loss_at_te + signal_loss_end) / 2
    else:
        avg_signal_loss = signal_loss_at_te
    
    # T2* decay causes underestimation of signal amplitude
    # This leads to underestimation of T1
    # Empirical correction based on signal loss
    # More signal loss -> more T1 underestimation
    
    # Bias factor: how much T1 is underestimated
    # Based on Robson et al. MRM 2013 for MOLLI with T2* effects
    signal_loss_fraction = 1 - avg_signal_loss
    
    # Empirical relationship: ~2-3% T1 underestimation per 10% signal loss
    t1_underestimation_fraction = 0.25 * signal_loss_fraction
    
    # Correction factor to compensate for underestimation
    bias_correction = 1.0 / (1.0 - t1_underestimation_fraction)
    
    # Limit correction to reasonable range
    bias_correction = np.clip(bias_correction, 1.0, 1.2)
    
    return bias_correction

def apply_molli_t1_corrections(t1_apparent, a, b, flip_angle, tr, te=None, 
                               t2star=None, heart_rate=None,
                               apply_look_locker=True, 
                               apply_t2star_bias=False,
                               apply_heart_rate=False,
                               apply_b1_correction=False, 
                               b1_factor=1.0,
                               apply_cardiac_specific=False):
    """
    Apply comprehensive T1 corrections for cardiac MOLLI sequences
    Based on current literature recommendations
    
    Parameters:
    -----------
    t1_apparent : float or array
        Apparent T1 from fit (ms)
    a, b : float
        Fitting parameters from IR model
    flip_angle : float
        Readout flip angle (degrees)
    tr : float
        Repetition time (ms)
    te : float, optional
        Echo time (ms) - for T2* bias estimation
    t2star : float, optional
        T2* value in ms
    heart_rate : float, optional
        Heart rate in bpm
    apply_look_locker : bool
        Apply Look-Locker correction
    apply_t2star_bias : bool
        Apply T2* bias correction
    apply_heart_rate : bool
        Apply heart rate correction
    apply_b1_correction : bool
        Apply B1 inhomogeneity correction
    b1_factor : float or array
        B1 field scaling factor (1.0 = nominal)
    apply_cardiac_specific : bool
        Apply cardiac-specific empirical correction
    
    Returns:
    --------
    t1_corrected : float or array
        Corrected T1 value(s)
    correction_info : dict
        Information about applied corrections
    """
    import numpy as np
    
    t1_corrected = t1_apparent
    correction_info = {
        'look_locker': 1.0,
        't2star_bias': 1.0,
        'heart_rate': 1.0,
        'b1': 1.0,
        'cardiac_specific': 1.0,
        'total': 1.0
    }
    
    # 1. Look-Locker correction (most important for MOLLI)
    if apply_look_locker and flip_angle is not None and tr is not None:
        ll_factor = calculate_look_locker_correction_factor(
            t1_apparent, flip_angle, tr, model_type='MOLLI'
        )
        t1_corrected = t1_corrected * ll_factor
        correction_info['look_locker'] = ll_factor
    
    # 2. T2* bias correction
    if apply_t2star_bias and te is not None and t2star is not None:
        t2star_correction = estimate_t2star_bias_in_molli(te, t2star)
        t1_corrected = t1_corrected * t2star_correction
        correction_info['t2star_bias'] = t2star_correction
    
    # 3. Heart rate correction (important for cardiac MOLLI)
    if apply_heart_rate and heart_rate is not None:
        # Based on Roujol et al. JCMR 2014
        # T1 error ≈ 2.8 ms per 10 bpm increase from baseline
        hr_reference = 60  # Reference heart rate
        
        # Calculate correction
        hr_difference = heart_rate - hr_reference
        t1_error_ms = 2.8 * (hr_difference / 10)
        
        # Apply as multiplicative correction
        hr_correction = 1 + (t1_error_ms / t1_apparent)
        hr_correction = np.clip(hr_correction, 0.9, 1.1)  # Limit to ±10%
        
        t1_corrected = t1_corrected * hr_correction
        correction_info['heart_rate'] = hr_correction
    
    # 4. B1 correction
    if apply_b1_correction and b1_factor != 1.0:
        # B1 affects both flip angle and inversion efficiency
        # Simplified correction - more sophisticated methods exist
        
        # Flip angle deviation effect
        flip_angle_factor = 1.0 + 0.2 * (1.0 - b1_factor)
        
        # Inversion efficiency effect (more significant)
        inversion_factor = 1.0 / b1_factor  # Approximate
        
        # Combined B1 correction
        b1_correction = 0.3 * flip_angle_factor + 0.7 * inversion_factor
        b1_correction = np.clip(b1_correction, 0.8, 1.2)
        
        t1_corrected = t1_corrected * b1_correction
        correction_info['b1'] = b1_correction
    
    # 5. Cardiac-specific empirical correction
    if apply_cardiac_specific:
        # Literature-based empirical factor for cardiac tissue
        # This accounts for various cardiac-specific effects not captured above
        cardiac_factor = 1.12  # 12% increase based on phantom/in-vivo comparisons
        t1_corrected = t1_corrected * cardiac_factor
        correction_info['cardiac_specific'] = cardiac_factor
    
    # Ensure physiological bounds
    if isinstance(t1_corrected, np.ndarray):
        t1_corrected = np.clip(t1_corrected, 100, 3000)
    else:
        t1_corrected = max(100, min(3000, t1_corrected))
    
    # Calculate total correction
    correction_info['total'] = np.prod([v for k, v in correction_info.items() if k != 'total'])
    
    return t1_corrected, correction_info

# Helper function for future T2 mapping support
def estimate_t2_from_multiecho(echo_times, signals, t2_guess=50):
    """
    Estimate T2 from multi-echo data
    For future T2 mapping functionality
    
    Parameters:
    -----------
    echo_times : array
        Echo times in ms
    signals : array
        Signal intensities at each echo time
    t2_guess : float
        Initial guess for T2 in ms
    
    Returns:
    --------
    t2 : float
        Estimated T2 value
    r_squared : float
        Goodness of fit
    """
    import numpy as np
    from scipy.optimize import curve_fit
    
    def t2_decay(te, s0, t2):
        return s0 * np.exp(-te / t2)
    
    try:
        # Ensure positive signals
        signals = np.abs(signals)
        
        # Initial parameters
        s0_guess = np.max(signals)
        p0 = [s0_guess, t2_guess]
        
        # Fit exponential decay
        popt, _ = curve_fit(t2_decay, echo_times, signals, 
                           p0=p0, bounds=([0, 1], [np.inf, 200]))
        
        s0_fit, t2_fit = popt
        
        # Calculate R-squared
        fitted = t2_decay(echo_times, s0_fit, t2_fit)
        residuals = signals - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((signals - np.mean(signals))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return t2_fit, r_squared
        
    except:
        return t2_guess, 0.0
    
def load_ir_dicoms_robust(paths):
    """Enhanced DICOM loading with better error handling and validation"""
    if not paths:
        raise ValueError("No files selected")
    
    valid_paths = []
    invalid_files = []
    
    # First pass - validate files
    for path in paths:
        try:
            ds = pydicom.dcmread(path)
            # Check if file has InversionTime tag
            if not hasattr(ds, 'InversionTime'):
                invalid_files.append((path, "Missing InversionTime tag"))
            else:
                valid_paths.append(path)
        except Exception as e:
            invalid_files.append((path, str(e)))
    
    if not valid_paths:
        raise ValueError("No valid inversion recovery DICOM files found")
    
    # Report invalid files if any
    if invalid_files:
        warning_msg = f"Warning: {len(invalid_files)} files were skipped:\n"
        warning_msg += "\n".join([f"- {os.path.basename(p)}: {e}" for p, e in invalid_files[:5]])
        if len(invalid_files) > 5:
            warning_msg += f"\n... and {len(invalid_files) - 5} more"
        messagebox.showwarning("File Loading Warning", warning_msg)
    
    # Sort by inversion time
    try:
        paths_sorted = sorted(valid_paths, key=lambda p: float(pydicom.dcmread(p).InversionTime))
    except Exception as e:
        raise ValueError(f"Error sorting files by inversion time: {str(e)}")
    
    # Load images and extract parameters
    images, tis = [], []
    tr_values, flip_angles = [], []
    te_values = []
    
    # Check for manufacturer-specific scaling
    # Read the first file to check manufacturer and possibly other tags
    first_ds = pydicom.dcmread(paths_sorted[0])
    manufacturer = getattr(first_ds, 'Manufacturer', '').lower()
    
    # Initialize scaling parameters
    ti_scaling_factor = 1.0  # Default: no scaling
    
    # Check for manufacturer-specific scaling factors
    if 'siemens' in manufacturer:
        # Siemens might use specific scaling for inversion times
        messagebox.showinfo("Scanner Info", 
                          f"Detected {manufacturer} scanner. InversionTime values will be used as provided.")
    elif 'ge' in manufacturer:
        # GE often stores times in centiseconds rather than milliseconds
        if hasattr(first_ds, 'InversionTime') and float(first_ds.InversionTime) < 100:
            ti_scaling_factor = 10.0  # Convert from centiseconds to milliseconds
            messagebox.showinfo("Scanner Info", 
                              f"Detected {manufacturer} scanner with possible time unit difference.\n"
                              f"Applying scaling factor of {ti_scaling_factor} to Inversion Times.")
    elif 'philips' in manufacturer:
        # Philips might use specific scaling
        messagebox.showinfo("Scanner Info", 
                          f"Detected {manufacturer} scanner. InversionTime values will be used as provided.")
    
    # Check if InversionTime units don't match expectations (too small or large)
    first_ti = float(first_ds.InversionTime)
    if first_ti > 0 and first_ti < 1:
        # If TI is a fraction of a millisecond, might be in seconds
        ti_scaling_factor = 1000.0
        messagebox.showinfo("Unit Conversion", 
                          f"InversionTime values appear to be in seconds.\n"
                          f"Converting to milliseconds (factor: {ti_scaling_factor}).")
    elif first_ti > 5000:
        # If TI is unusually large, might be in microseconds
        ti_scaling_factor = 0.001
        messagebox.showinfo("Unit Conversion", 
                          f"InversionTime values appear to be in microseconds.\n"
                          f"Converting to milliseconds (factor: {ti_scaling_factor}).")
    
    for path in paths_sorted:
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            
            # Apply rescale if available
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            img = img * slope + intercept
            
            # Extract TR (Repetition Time) if available
            if hasattr(ds, 'RepetitionTime'):
                tr_values.append(float(ds.RepetitionTime))
            else:
                tr_values.append(None)
            
            # Extract TE (Echo Time) if available
            if hasattr(ds, 'EchoTime'):
                te_values.append(float(ds.EchoTime))
            else:
                te_values.append(None)
                
            # Extract Flip Angle if available
            if hasattr(ds, 'FlipAngle'):
                flip_angles.append(float(ds.FlipAngle))
            else:
                flip_angles.append(None)
            
            # Store data with scaled InversionTime
            images.append(img)
            tis.append(float(ds.InversionTime) * ti_scaling_factor)
        except Exception as e:
            raise ValueError(f"Error loading {os.path.basename(path)}: {str(e)}")
    
    # Validate we have enough images
    if len(images) < 3:
        raise ValueError(f"At least 3 images required for T1 fitting, but only {len(images)} valid images found")
    
    # Check that all images have the same dimensions
    shapes = [img.shape for img in images]
    if len(set(shapes)) > 1:
        raise ValueError(f"All images must have the same dimensions. Found: {set(shapes)}")
    
    # Use the first non-None value for consistency
    tr_value = next((x for x in tr_values if x is not None), None)
    flip_value = next((x for x in flip_angles if x is not None), None)
    te_value = next((x for x in te_values if x is not None), None)
    
    # Store sequence parameters
    sequence_params = {
        'TR': tr_value,
        'FlipAngle': flip_value,
        'TI_ScalingFactor': ti_scaling_factor,
        'Manufacturer': manufacturer,
        'TE': te_value
    }
    
    return np.stack(images, axis=0), np.array(tis), sequence_params

def calculate_inversion_efficiency(flip_angle, b1_factor=None):
    """
    Calculate inversion efficiency based on flip angle
    
    Parameters:
    -----------
    flip_angle : float
        Nominal flip angle in degrees
    b1_factor : float or ndarray, optional
        B1 map scale factor (1.0 = perfect B1)
    
    Returns:
    --------
    efficiency : float or ndarray
        Inversion efficiency factor (0-1)
    """
    if flip_angle is None:
        # Default to perfect inversion if flip angle not available
        return 1.0
    
    # Convert flip angle to radians (assuming it's the inversion pulse angle)
    flip_angle_rad = np.abs(flip_angle) * np.pi / 180.0
    
    # Apply B1 correction if available
    if b1_factor is not None:
        # Scale the flip angle based on B1 inhomogeneity
        flip_angle_rad = flip_angle_rad * b1_factor
    
    # For a nominal 180° inversion pulse, efficiency is related to cos(flip_angle)
    # This is a simplified model - real efficiency depends on pulse shape, B1 inhomogeneity, etc.
    efficiency = (1 - np.cos(flip_angle_rad)) / 2
    
    # Ensure efficiency is in valid range
    if isinstance(efficiency, np.ndarray):
        efficiency = np.clip(efficiency, 0.0, 1.0)
    else:
        efficiency = max(0.0, min(1.0, efficiency))
    
    return efficiency

def simple_t1_fit(signal, tis):
    """
    Simple T1 fitting for cases where the robust method fails
    Uses basic three-parameter fitting with good initial guesses
    """
    # Apply basic IR physics constraints
    signal_copy = signal.copy()
    sort_indices = np.argsort(tis)
    sorted_tis = tis[sort_indices]
    sorted_signal = signal_copy[sort_indices]
    
    # Ensure first points are negative, later points positive
    sorted_signal[0] = -abs(sorted_signal[0])
    if len(sorted_signal) > 1:
        sorted_signal[1] = -abs(sorted_signal[1])
    for i in range(3, len(sorted_signal)):
        sorted_signal[i] = abs(sorted_signal[i])
    
    # Third point can be either positive or negative - try both
    # First with positive
    try:
        if len(sorted_signal) > 2:
            sorted_signal[2] = abs(sorted_signal[2])
        
        popt, _ = opt.curve_fit(
            ir_model_simple, sorted_tis, sorted_signal,
            p0=[np.max(sorted_signal), 2*np.ptp(sorted_signal), 1000],
            bounds=([0, -np.inf, 10], [np.inf, np.inf, 5000]),
            maxfev=200,
            method='trf'
        )
        
        # Calculate fit quality
        fit = ir_model_simple(sorted_tis, *popt)
        residuals = sorted_signal - fit
        ss_res = np.sum(residuals**2)
        pos_error = ss_res
        pos_result = popt
        pos_fit = fit
    except:
        pos_error = np.inf
        pos_result = None
        pos_fit = None
    
    # Then with negative
    try:
        if len(sorted_signal) > 2:
            sorted_signal[2] = -abs(sorted_signal[2])
        
        popt, _ = opt.curve_fit(
            ir_model_simple, sorted_tis, sorted_signal,
            p0=[np.max(sorted_signal), 2*np.ptp(sorted_signal), 1000],
            bounds=([0, -np.inf, 10], [np.inf, np.inf, 5000]),
            maxfev=200,
            method='trf'
        )
        
        # Calculate fit quality
        fit = ir_model_simple(sorted_tis, *popt)
        residuals = sorted_signal - fit
        ss_res = np.sum(residuals**2)
        neg_error = ss_res
        neg_result = popt
        neg_fit = fit
    except:
        neg_error = np.inf
        neg_result = None
        neg_fit = None
    
    # Choose the better fit
    if pos_error <= neg_error and pos_result is not None:
        best_result = pos_result
        best_fit = pos_fit
    elif neg_result is not None:
        best_result = neg_result
        best_fit = neg_fit
    else:
        # Both failed, use defaults
        a = np.max(sorted_signal)
        b = np.ptp(sorted_signal)
        t1 = 1000.0
        best_result = (a, b, t1)
        best_fit = ir_model_simple(sorted_tis, a, b, t1)
    
    # Unsort back to original order
    unsorted_signal = sorted_signal.copy()
    unsorted_fit = np.zeros_like(signal)
    
    for i, idx in enumerate(sort_indices):
        unsorted_fit[idx] = best_fit[i]
        unsorted_signal[idx] = sorted_signal[i]
    
    # Calculate R-squared for fit quality assessment
    residuals = unsorted_signal - unsorted_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((unsorted_signal - np.mean(unsorted_signal))**2)
    r_squared = 1 - (ss_res / ss_tot if ss_tot != 0 else 0)
    
    # Create fit status dictionary
    fit_status = {
        'model_used': 'simple',
        'r_squared': r_squared,
        'fallback_used': True
    }
    
    return best_result, unsorted_signal, unsorted_fit, fit_status

def physically_constrained_t1_fit(signal, tis, flip_angle=None, tr=None, te=None, t2star=40, 
                                     inversion_efficiency=1.0, timeout=1.0, 
                                     model='simple', correction_method='full',
                                     apply_cardiac_correction=False,
                                     apply_mt_correction=False, 
                                     mt_fraction=None):  
        """
        T1 fitting with strict enforcement of physical constraints for GRE MOLLI sequences
        
        Parameters:
        -----------
        signal : ndarray
            Signal values
        tis : ndarray
            Inversion times in ms
        flip_angle : float, optional
            Flip angle in degrees
        tr : float, optional
            Repetition time in ms
        te : float, optional
            Echo time in ms
        t2star : float, optional
            T2* value in ms, default is 40 ms for myocardium
        inversion_efficiency : float, optional
            Inversion efficiency (0-1)
        timeout : float, optional
            Timeout for fitting in seconds
        model : str, optional
            Fitting model to use: 'simple' or 'classic'
        correction_method : str, optional
            Correction method to use: 'look_locker', 'full', or 'none'
        apply_cardiac_correction : bool, optional
            Whether to apply cardiac-specific correction factor
        
        Returns:
        --------
        popt : tuple
            Fitted parameters (a, b, t1)
        processed_signal : ndarray
            Processed signal with appropriate sign adjustments
        fitted_curve : ndarray
            Fitted curve values
        fit_status : dict
            Information about the fit quality and procedure
        """
        import time
        import numpy as np
        import scipy.optimize as opt
        
        # First, sort by TI values to ensure correct temporal order
        sort_indices = np.argsort(tis)
        sorted_tis = tis[sort_indices]
        sorted_signal = signal[sort_indices].copy()
        
        # Determine number of inversion times
        n_tis = len(sorted_tis)
        if n_tis < 3:
            raise ValueError("At least 3 inversion times are required for reliable fitting")
        
        # Initialize fit status tracking
        fit_status = {
            'model_used': model,
            'correction_used': correction_method,
            'r_squared': None,
            'rmse': None,
            'time_taken': None,
            'iterations': 0,
            'sign_flips_tested': 0,
            'fallback_used': False,
            'status': 'initialized'
        }
        
        # Select model function
        if model.lower() == 'classic':
            model_func = lambda ti, a, b, t1: a * (1 - 2 * b * inversion_efficiency * np.exp(-ti / t1))
        else:  # Simple model
            model_func = lambda ti, a, b, t1: a - b * inversion_efficiency * np.exp(-ti / t1)
        
        # Physical constraints based on typical MOLLI recovery:
        # For most biological tissue, first point should be negative (closest to inversion)
        sorted_signal[0] = -abs(sorted_signal[0])
        
        # Fix the sign pattern to the most common expected pattern 
        # (negative initial points transitioning to positive later points)
        # This avoids testing multiple sign patterns which is computationally expensive
        sign_pattern = np.ones(n_tis)
        
        # First point(s) negative, rest positive - common pattern for tissues with T1 ~1000ms
        sign_threshold = min(2, n_tis-1)  # Try only first two negative
        sign_pattern[:sign_threshold] = -1
        
        # Apply the sign pattern
        for i in range(n_tis):
            sorted_signal[i] = abs(sorted_signal[i]) * sign_pattern[i]
        
        # Define reasonable initial guesses focused around T1 = 1000 ms
        signal_max = np.max(abs(sorted_signal))
        signal_range = np.ptp(sorted_signal)
        
        # Use fewer, more targeted initial guesses
        if model.lower() == 'classic':
            # Only two guesses: one at 1000ms and one slightly higher
            initial_guesses = [
                (signal_max, 1.0, 1000),  # Main guess at 1000ms with b=1.0 (perfect inversion)
                (signal_max, 0.8, 1200)    # Secondary guess wit slightly imperfect inversion
            ]
            bounds = ([0, 0.5, 500], [np.inf, 1.5, 1800])  # Narrower bounds around physiological range
        else:  # Simple model
            # Only two guesses: one at 1000ms and one slightly higher
            initial_guesses = [
                (signal_max, signal_range * 1.5, 1000),  # Main guess at 1000ms
                (signal_max, signal_range * 1.8, 1200)   # Secondary guess
            ]
            bounds = ([0, 0, 500], [np.inf, np.inf, 1800])  # Narrower bounds  for b around physiological range
        
        # Try initial guesses with reduced iterations
        best_error = np.inf
        best_result = None
        best_processed_signal = None
        best_fitted_curve = None
        
        start_time = time.time()
        total_iterations = 0
        
        for guess in initial_guesses:
            # Check timeout
            if time.time() - start_time > timeout:
                fit_status['status'] = 'timeout'
                break
            
            total_iterations += 1
            try:
                popt, pcov = opt.curve_fit(
                    model_func,
                    sorted_tis, 
                    sorted_signal, 
                    p0=guess,
                    bounds=bounds,
                    maxfev=200,  # Reduced from 500 for speed
                    method='trf'
                )
                
                # Calculate fit quality
                fitted = model_func(sorted_tis, *popt)
                residuals = sorted_signal - fitted
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((sorted_signal - np.mean(sorted_signal))**2)
                r_squared = 1 - (ss_res / ss_tot if ss_tot != 0 else 0)
                rmse = np.sqrt(np.mean(residuals**2))
                
                # Use combination of R² and RMSE for error metric
                error = (1 - r_squared) + 0.1 * (rmse / signal_max)
                
                if error < best_error:
                    best_error = error
                    # Store raw parameters before correction
                    a, b, t1_apparent = popt
                    
                    # Initialize t1_corrected to t1_apparent as a default
                    t1_corrected = t1_apparent

                    # FIRST: Apply the standard corrections (Look-Locker, etc.)
                    try:
                        if correction_method.startswith('custom:'):
                            # Extract which corrections to apply from the string
                            correction_codes = correction_method[7:]  # Remove 'custom:' prefix
                            apply_ll = 'L' in correction_codes
                            apply_t2star = 'T' in correction_codes
                            apply_sat = 'S' in correction_codes
                            apply_inv = 'B' in correction_codes
                            
                            # Get T2* value from UI if available (you'll need to pass this as parameter)
                            # For now, using default
                            t2star_value = t2star if t2star is not None else 40  # Default myocardial T2*
                            
                            # Use enhanced correction function with proper T2* bias correction
                            t1_corrected, correction_factors = enhanced_t1_correction_gre_flexible(
                                t1_apparent, a, b, 
                                flip_angle, tr, te, t2star_value,
                                inversion_efficiency,
                                apply_look_locker=apply_ll,
                                apply_t2star=apply_t2star,
                                apply_saturation=apply_sat,
                                apply_inversion=apply_inv,
                                apply_cardiac_correction=apply_cardiac_correction
                            )
                            
                        elif correction_method.lower() == 'molli_standard':
                            # Use the new MOLLI-specific correction function
                            t1_corrected, correction_info = apply_molli_t1_corrections(
                                t1_apparent, a, b,
                                flip_angle, tr, te,
                                t2star=t2star,
                                heart_rate=None,  # You could add heart rate input to UI
                                apply_look_locker=True,
                                apply_t2star_bias=True,
                                apply_heart_rate=False,
                                apply_b1_correction=False,
                                b1_factor=1.0,
                                apply_cardiac_specific=apply_cardiac_correction
                            )
                            # Convert correction_info dict to correction_factors format
                            correction_factors = {
                                'look_locker': correction_info.get('look_locker', 1.0),
                                't2star_bias': correction_info.get('t2star_bias', 1.0),
                                'saturation': 1.0,
                                'inversion': 1.0,
                                'cardiac': correction_info.get('cardiac_specific', 1.0),
                                'total': correction_info.get('total', 1.0)
                            }
                        
                        elif correction_method.lower() == 'look_locker_t2star':
                            # Custom combination of Look-Locker and T2* corrections
                            t1_corrected, correction_factors = enhanced_t1_correction_gre_flexible(
                                t1_apparent, a, b,
                                flip_angle, tr, te, t2star,
                                inversion_efficiency,
                                apply_look_locker=True,
                                apply_t2star=True,
                                apply_saturation=False,
                                apply_inversion=False
                            )
                        elif correction_method.lower() == 'look_locker_inversion':
                            # Custom combination of Look-Locker and inversion corrections
                            t1_corrected, correction_factors = enhanced_t1_correction_gre_flexible(
                                t1_apparent, a, b,
                                flip_angle, tr, te, t2star,
                                inversion_efficiency,
                                apply_look_locker=True,
                                apply_t2star=False,
                                apply_saturation=False,
                                apply_inversion=True
                            )
                        elif correction_method.lower() == 'standard_gre':
                            # Standard GRE corrections: LL + saturation
                            t1_corrected, correction_factors = enhanced_t1_correction_gre_flexible(
                                t1_apparent, a, b,
                                flip_angle, tr, te, t2star,
                                inversion_efficiency,
                                apply_look_locker=True,
                                apply_t2star=False,
                                apply_saturation=True,
                                apply_inversion=False
                            )
                        else:
                            # Use the enhanced_t1_correction_gre function for other methods
                            t1_corrected, correction_factors = enhanced_t1_correction_gre(
                                t1_apparent, a, b, 
                                flip_angle, tr, te, t2star, 
                                inversion_efficiency,
                                correction_method=correction_method
                            )
                    except Exception as corr_error:
                        # If correction fails, use uncorrected value
                        print(f"Correction error: {corr_error}")
                        t1_corrected = t1_apparent
                        correction_factors = {'total': 1.0}
                    
                    # SECOND: Now apply MT correction if requested (AFTER t1_corrected exists)
                    if apply_mt_correction:
                        try:
                            mt_detected, mt_score, p_val = detect_mt_from_residuals(
                                sorted_tis, sorted_signal, fitted, threshold=0.15
                            )
                            
                            # Store MT info in fit_status
                            fit_status['mt_detected'] = mt_detected
                            fit_status['mt_score'] = mt_score
                            
                            # Estimate MT parameters if detected
                            if mt_detected or mt_score > 0.1:
                                if mt_fraction is None:
                                    mt_fraction_est, k_exchange = estimate_mt_fraction(
                                        sorted_tis, residuals, sorted_signal, method='enhanced'
                                    )
                                else:
                                    mt_fraction_est = mt_fraction
                                    k_exchange = 20.0
                                
                                # Calculate MT correction factor (with field strength consideration)
                                mt_correction_factor = calculate_mt_correction_factor(
                                    mt_fraction_est, k_exchange, sorted_tis, tr, flip_angle,
                                    tissue_type='cardiac', field_strength=1.5  # Specify 1.5T
                                )
                                
                                # Store MT info
                                fit_status['mt_fraction'] = mt_fraction_est
                                fit_status['mt_correction_factor'] = mt_correction_factor
                            else:
                                # Apply minimal baseline MT correction for 1.5T
                                fit_status['mt_fraction'] = 0.05  # Lower for 1.5T
                                fit_status['mt_correction_factor'] = 1.03  # Much smaller correction
                            
                            # Apply MT correction to the already-corrected T1
                            t1_corrected = t1_corrected * fit_status['mt_correction_factor']
                            
                            # Update correction factors
                            if 'mt' not in correction_factors:
                                correction_factors['mt'] = fit_status['mt_correction_factor']
                            correction_factors['total'] = correction_factors['total'] * fit_status['mt_correction_factor']
                        except Exception as mt_error:
                            # If MT correction fails, continue without it
                            print(f"MT correction error: {mt_error}")
                            fit_status['mt_detected'] = False
                            fit_status['mt_fraction'] = 0.0
                            fit_status['mt_correction_factor'] = 1.0
                    else:
                        # No MT correction requested
                        fit_status['mt_detected'] = False
                        fit_status['mt_fraction'] = 0.0
                        fit_status['mt_correction_factor'] = 1.0
                    
                    # Store the final result
                    best_result = (a, b, t1_corrected)
                    best_processed_signal = sorted_signal.copy()
                    best_fitted_curve = fitted
                    fit_status['r_squared'] = r_squared
                    fit_status['rmse'] = rmse
                    fit_status['correction_factors'] = correction_factors
                    
                    # If we found a good fit, stop early
                    if r_squared > 0.95:
                        fit_status['status'] = 'early_stop_good_fit'
                        break
                        
            except Exception as e:
                # Just continue with next guess if this one fails
                print(f"Fitting iteration error: {e}")
                continue
        
        fit_status['time_taken'] = time.time() - start_time
        fit_status['iterations'] = total_iterations
        
        # If we couldn't find a good fit, use a simpler approach
        if best_result is None:
            try:
                # Fallback to a simpler approach with direct fitting
                sorted_signal[0] = -abs(sorted_signal[0])  # Always negative
                if n_tis > 1:
                    sorted_signal[1] = -abs(sorted_signal[1])  # Always negative
                
                # For all remaining points, use positive values
                for i in range(2, n_tis):
                    sorted_signal[i] = abs(sorted_signal[i])
                
                # Use a direct approach with conservative parameters
                if model.lower() == 'classic':
                    model_simple = lambda ti, a, b, t1: a * (1 - 2 * b * inversion_efficiency * np.exp(-ti / t1))
                    p0 = [signal_max, 0.5, 1000]  # Direct guess centered at 1000ms
                    bounds = ([0, 0, 500], [np.inf, 1.0, 1800])
                else:
                    model_simple = lambda ti, a, b, t1: a - b * inversion_efficiency * np.exp(-ti / t1)
                    p0 = [signal_max, signal_range, 1000]  # Direct guess centered at 1000ms
                    bounds = ([0, 0, 500], [np.inf, np.inf, 2500])
                
                popt, _ = opt.curve_fit(
                    model_simple, 
                    sorted_tis, 
                    sorted_signal,
                    p0=p0,
                    bounds=bounds,
                    maxfev=300,  # Slightly more iterations for fallback
                    method='trf'
                )
                
                # Calculate fit quality
                fitted = model_simple(sorted_tis, *popt)
                residuals = sorted_signal - fitted
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((sorted_signal - np.mean(sorted_signal))**2)
                r_squared = 1 - (ss_res / ss_tot if ss_tot != 0 else 0)
                rmse = np.sqrt(np.mean(residuals**2))
                
                # Apply correction to T1 with support for custom methods
                a, b, t1_apparent = popt
                
                # Initialize t1_corrected to t1_apparent as default
                t1_corrected = t1_apparent
                
                try:
                    if correction_method.startswith('custom:'):
                        correction_codes = correction_method[7:]
                        apply_ll = 'L' in correction_codes
                        apply_t2star = 'T' in correction_codes
                        apply_sat = 'S' in correction_codes
                        apply_inv = 'B' in correction_codes
                        
                        t1_corrected, correction_factors = enhanced_t1_correction_gre_flexible(
                            t1_apparent, a, b, 
                            flip_angle, tr, te, t2star, 
                            inversion_efficiency,
                            apply_look_locker=apply_ll,
                            apply_t2star=apply_t2star,
                            apply_saturation=apply_sat,
                            apply_inversion=apply_inv
                        )
                    elif correction_method.lower() == 'look_locker_t2star':
                        t1_corrected, correction_factors = enhanced_t1_correction_gre_flexible(
                            t1_apparent, a, b,
                            flip_angle, tr, te, t2star,
                            inversion_efficiency,
                            apply_look_locker=True,
                            apply_t2star=True,
                            apply_saturation=False,
                            apply_inversion=False
                        )
                    elif correction_method.lower() == 'look_locker_inversion':
                        t1_corrected, correction_factors = enhanced_t1_correction_gre_flexible(
                            t1_apparent, a, b,
                            flip_angle, tr, te, t2star,
                            inversion_efficiency,
                            apply_look_locker=True,
                            apply_t2star=False,
                            apply_saturation=False,
                            apply_inversion=True
                        )
                    elif correction_method.lower() == 'standard_gre':
                        t1_corrected, correction_factors = enhanced_t1_correction_gre_flexible(
                            t1_apparent, a, b,
                            flip_angle, tr, te, t2star,
                            inversion_efficiency,
                            apply_look_locker=True,
                            apply_t2star=False,
                            apply_saturation=True,
                            apply_inversion=False
                        )
                    else:
                        # Use the enhanced_t1_correction_gre function for other methods
                        t1_corrected, correction_factors = enhanced_t1_correction_gre(
                            t1_apparent, a, b, 
                            flip_angle, tr, te, t2star, 
                            inversion_efficiency,
                            correction_method=correction_method
                        )
                except Exception as corr_error:
                    # If correction fails in fallback, use uncorrected value
                    print(f"Fallback correction error: {corr_error}")
                    t1_corrected = t1_apparent
                    correction_factors = {'total': 1.0}
                
                best_result = (a, b, t1_corrected)
                best_processed_signal = sorted_signal.copy()
                best_fitted_curve = fitted
                fit_status['r_squared'] = r_squared
                fit_status['rmse'] = rmse
                fit_status['fallback_used'] = True
                fit_status['correction_factors'] = correction_factors
                fit_status['status'] = 'fallback_success'
                
            except Exception as e:
                # Last resort: use default values
                print(f"Fallback fitting error: {e}")
                a = signal_max
                b = signal_range
                t1 = 1000.0  # Default in ms
                
                best_result = (a, b, t1)
                best_processed_signal = sorted_signal.copy()
                best_fitted_curve = model_func(sorted_tis, a, b, t1)
                
                fit_status['fallback_used'] = True
                fit_status['status'] = 'default_values'
                fit_status['correction_factors'] = {'total': 1.0}
        
        # Convert back to original order
        unsorted_processed_signal = np.zeros_like(signal)
        unsorted_fitted_curve = np.zeros_like(signal)
        
        # Make sure best_result exists
        if best_result is None:
            # Ultimate fallback
            a = np.max(abs(signal))
            b = np.ptp(signal)
            t1 = 1000.0
            best_result = (a, b, t1)
            best_processed_signal = sorted_signal.copy()
            best_fitted_curve = model_func(sorted_tis, a, b, t1)
        
        a, b, t1 = best_result
        
        # Unsort the signals back to original order
        for i, idx in enumerate(sort_indices):
            unsorted_processed_signal[idx] = best_processed_signal[i]
            unsorted_fitted_curve[idx] = model_func(tis[idx], a, b, t1)
        
        # Apply cardiac correction if requested and not already applied
        if apply_cardiac_correction and 'cardiac' not in fit_status.get('correction_factors', {}):
            # Literature-based empirical correction for cardiac tissue
            cardiac_factor = 1.12  # Additional 12% increase
            t1 = t1 * cardiac_factor
            best_result = (a, b, t1)
            
            # Add to correction factors
            if 'correction_factors' not in fit_status:
                fit_status['correction_factors'] = {}
                
            fit_status['correction_factors']['cardiac'] = cardiac_factor
            
            # Update total correction factor
            if 'total' in fit_status['correction_factors']:
                fit_status['correction_factors']['total'] *= cardiac_factor
            else:
                fit_status['correction_factors']['total'] = cardiac_factor
        
        # Return the values
        return best_result, unsorted_processed_signal, unsorted_fitted_curve, fit_status

def apply_enhanced_mt_correction(self):
    """Apply enhanced MT correction to the entire T1 map"""
    if not hasattr(self, 't1_map') or self.t1_map is None:
        messagebox.showinfo("Info", "Calculate T1 first")
        return
    
    try:
        self.update_status("Applying enhanced MT correction...")
        
        # Identify septal region
        septal_mask = identify_septal_region(self.t1_map, percentile_threshold=15)
        
        # Get ROI data
        x1, x2, y1, y2 = self.roi_coords
        roi_stack = self.corrected_stack[:, y1:y2, x1:x2]
        
        # Analyze MT effects with enhanced detection
        correction_map, confidence_map, mt_stats = analyze_ti_signals_for_mt(
            self.tis, roi_stack, enhanced_detection=True
        )
        
        # Apply correction with special septal handling
        self.t1_map_mt_corrected = apply_mt_correction_to_t1map(
            self.t1_map, 
            mt_map=correction_map,
            septal_mask=septal_mask,
            septal_boost=1.15  # Additional 15% for septum
        )
        
        # Update display
        self.t1_map = self.t1_map_mt_corrected
        
        # Show statistics
        mean_correction = mt_stats['mean_correction_factor']
        detected_fraction = mt_stats['detected_fraction'] * 100
        
        messagebox.showinfo("MT Correction Applied",
                          f"Enhanced MT correction applied:\n"
                          f"Mean correction factor: {mean_correction:.2f}\n"
                          f"MT effects detected in: {detected_fraction:.1f}% of pixels\n"
                          f"Septal region received additional correction")
        
        # Update visualization
        self.show_t1_map_with_overlay()
        self.update_status("Enhanced MT correction complete")
        
    except Exception as e:
        self.update_status("Error in MT correction")
        show_error_dialog("MT Correction Error", "Error applying enhanced MT correction", e)



# Add these functions to support B1 map loading and application

def load_and_process_b1_map(b1_file_path, reference_image=None):
    """
    Load and process a B1 map for T1 correction
    
    Parameters:
    -----------
    b1_file_path : str
        Path to the B1 map DICOM file
    reference_image : ndarray, optional
        Reference image to match dimensions and orientation
        
    Returns:
    --------
    b1_map : ndarrayif error < best_error:
        Processed B1 map with values normalized to 1.0 (where 1.0 = nominal field)
    """
    import numpy as np
    import pydicom
    import scipy.ndimage as ndi
    
    try:
        # Load the B1 map
        b1_ds = pydicom.dcmread(b1_file_path)
        b1_raw = b1_ds.pixel_array.astype(np.float32)
        
        # Apply rescale if available
        slope = float(getattr(b1_ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(b1_ds, 'RescaleIntercept', 0.0))
        b1_raw = b1_raw * slope + intercept
        
        # Check for vendor-specific scaling factors in private tags
        manufacturer = getattr(b1_ds, 'Manufacturer', '').lower()
        
        # Default scaling: assume values are relative to nominal B1 (1.0 = 100%)
        scaling_factor = 1.0
        
        # Vendor-specific processing
        if 'siemens' in manufacturer:
            # For Siemens, B1 maps are often stored as relative flip angle %
            # Convert from % to ratio (e.g., 80% -> 0.8)
            if np.max(b1_raw) > 5:  # Likely percentage units
                scaling_factor = 0.01
            
        elif 'ge' in manufacturer:
            # For GE, may need different processing
            # Typically stored as ratio already, but check range
            if np.max(b1_raw) > 5:  # Might be in different units
                scaling_factor = 0.01
                
        elif 'philips' in manufacturer:
            # For Philips, might be stored as relative B1 ratio already
            # But check if it needs normalization
            if np.max(b1_raw) > 5:  # Might be percentage
                scaling_factor = 0.01
        
        # Apply scaling
        b1_map = b1_raw * scaling_factor
        
        # Basic sanity checks on values
        if np.max(b1_map) < 0.1 or np.max(b1_map) > 10:
            print(f"Warning: B1 map values out of expected range: min={np.min(b1_map)}, max={np.max(b1_map)}")
            print("Attempting to normalize assuming relative units...")
            
            # Try to normalize based on median non-zero value
            nonzero_values = b1_map[b1_map > 0]
            if len(nonzero_values) > 0:
                med_value = np.median(nonzero_values)
                if med_value > 0:
                    # Normalize so median is around 1.0
                    b1_map = b1_map / med_value
        
        # Apply light smoothing to reduce noise
        b1_map = ndi.gaussian_filter(b1_map, sigma=1.0)
        
        # Ensure positive values and reasonable range
        b1_map = np.clip(b1_map, 0.1, 2.0)
        
        # If reference image provided, ensure B1 map matches it
        if reference_image is not None:
            if b1_map.shape != reference_image.shape:
                # Need to resample
                from scipy.interpolate import RegularGridInterpolator
                
                # Create coordinates for current B1 map
                y_b1, x_b1 = np.mgrid[0:b1_map.shape[0], 0:b1_map.shape[1]]
                y_b1 = y_b1 / (b1_map.shape[0] - 1)
                x_b1 = x_b1 / (b1_map.shape[1] - 1)
                
                # Create interpolator
                points = (np.linspace(0, 1, b1_map.shape[0]), np.linspace(0, 1, b1_map.shape[1]))
                interpolator = RegularGridInterpolator(points, b1_map, bounds_error=False, fill_value=1.0)
                
                # Create coordinates for target shape
                y_target, x_target = np.mgrid[0:reference_image.shape[0], 0:reference_image.shape[1]]
                y_target = y_target / (reference_image.shape[0] - 1)
                x_target = x_target / (reference_image.shape[1] - 1)
                
                # Interpolate
                pts = np.column_stack((y_target.flatten(), x_target.flatten()))
                b1_map = interpolator(pts).reshape(reference_image.shape)
        
        return b1_map
        
    except Exception as e:
        print(f"Error loading B1 map: {str(e)}")
        if reference_image is not None:
            # Return a uniform map matching reference image
            return np.ones_like(reference_image)
        else:
            # Return a small uniform map
            return np.ones((10, 10))

def apply_b1_correction_to_t1_map(t1_map, b1_map, flip_angle=None, tr=None, t2star=None, inversion_efficiency=None):
    """
    Apply B1 correction to an existing T1 map - FIXED ARRAY HANDLING
    
    Parameters:
    -----------
    t1_map : ndarray
        Uncorrected T1 map
    b1_map : ndarray
        B1 map with same dimensions as T1 map
    flip_angle : float, optional
        Nominal flip angle in degrees
    tr : float, optional
        Repetition time in ms
    t2star : float, optional
        T2* value in ms
    inversion_efficiency : float, optional
        Directly specified inversion efficiency (0-1)
        
    Returns:
    --------
    t1_corrected : ndarray
        B1-corrected T1 map
    correction_map : ndarray
        Map of correction factors applied
    """
    import numpy as np
    
    # Ensure b1_map and t1_map have the same dimensions
    if b1_map.shape != t1_map.shape:
        raise ValueError(f"B1 map shape {b1_map.shape} does not match T1 map shape {t1_map.shape}")
    
    # FIXED: Create explicit copies to avoid view issues
    t1_corrected = t1_map.copy()
    correction_map = np.ones_like(t1_map)
    
    # Get masks for valid T1 values and valid B1 values
    valid_t1 = ~np.isnan(t1_map)
    valid_b1 = (b1_map > 0) & (b1_map < 5)  # Reasonable B1 range
    
    # Only process where both are valid
    valid_mask = valid_t1 & valid_b1
    
    # Calculate correction factor for each pixel
    if valid_mask.any():
        # Extract valid values for processing
        valid_t1_values = t1_map[valid_mask].copy()
        valid_b1_values = b1_map[valid_mask].copy()
        
        if flip_angle is not None:
            # Calculate actual flip angle
            actual_flip = flip_angle * valid_b1_values
            
            # Approximate correction factor based on flip angle deviation
            angle_factor = 1 + 0.5 * (1 - valid_b1_values)
            
            # Modulation of inversion efficiency
            if inversion_efficiency is None:
                # Calculate from B1 map
                nominal_inv = np.radians(180)
                actual_inv = nominal_inv * valid_b1_values
                inv_efficiency = (1 - np.cos(actual_inv)) / 2
                
                # Correction factor for inversion efficiency
                inv_factor = 1.0 / np.clip(inv_efficiency, 0.1, 1.0)
                inv_factor = np.clip(inv_factor, 1.0, 2.0)
            else:
                # Use provided efficiency but scale with B1
                scaled_efficiency = inversion_efficiency * valid_b1_values
                scaled_efficiency = np.clip(scaled_efficiency, 0.1, 1.0)
                inv_factor = 1.0 / scaled_efficiency
                inv_factor = np.clip(inv_factor, 1.0, 2.0)
                
            # Combined correction (flip angle effect + inversion effect)
            combined_factor = angle_factor * 0.3 + inv_factor * 0.7
            
            # FIXED: Apply correction with explicit array handling
            correction_values = combined_factor.copy()
            corrected_t1_values = valid_t1_values * correction_values
            
            # Assign back to arrays
            correction_map[valid_mask] = correction_values
            t1_corrected[valid_mask] = corrected_t1_values
            
        else:
            # Simplified approach when flip angle unknown
            correction_values = 1.0 / valid_b1_values
            correction_values = np.clip(correction_values, 0.7, 1.3)
            corrected_t1_values = valid_t1_values * correction_values
            
            # Assign back to arrays
            correction_map[valid_mask] = correction_values
            t1_corrected[valid_mask] = corrected_t1_values
    
    # Constrain final values to physiological range
    t1_corrected = np.clip(t1_corrected, 100, 3000)
    
    return t1_corrected, correction_map

# Add these quality assessment functions

def create_comprehensive_quality_maps(t1_map, pixel_cache):
    """
    Create comprehensive quality assessment maps for T1 mapping
    
    Parameters:
    -----------
    t1_map : ndarray
        T1 map
    pixel_cache : dict
        Dictionary of pixel-wise fitting results
        
    Returns:
    --------
    quality_maps : dict
        Dictionary containing different quality maps:
        - r_squared: R² goodness of fit
        - rmse: Root mean square error 
        - confidence: T1 estimation confidence
        - precision: Estimated precision (inverse of uncertainty)
        - inversion_efficiency: Inversion efficiency map
        - correction_factor: Total correction factor applied
    """
    import numpy as np
    
    # Get dimensions from t1_map
    h, w = t1_map.shape
    
    # Initialize quality maps with NaN values
    r_squared_map = np.ones((h, w)) * np.nan
    rmse_map = np.ones((h, w)) * np.nan
    confidence_map = np.ones((h, w)) * np.nan
    precision_map = np.ones((h, w)) * np.nan
    inv_efficiency_map = np.ones((h, w)) * np.nan
    correction_map = np.ones((h, w)) * np.nan
    
    # Extract values from pixel cache
    for key, data in pixel_cache.items():
        x, y = key
        if 0 <= y < h and 0 <= x < w:  # Check bounds
            # R-squared (fit quality)
            if 'r_squared' in data and data['r_squared'] is not None:
                r_squared_map[y, x] = data['r_squared']
            
            # RMSE (fit error)
            if 'rmse' in data and data['rmse'] is not None:
                rmse_map[y, x] = data['rmse']
            elif 'signal' in data and 'fit' in data:
                # Calculate RMSE if not directly provided
                residuals = data['signal'] - data['fit']
                rmse_map[y, x] = np.sqrt(np.mean(residuals**2))
            
            # Store inversion efficiency if available
            if 'inversion_efficiency' in data:
                inv_efficiency_map[y, x] = data['inversion_efficiency']
            
            # Store correction factor if available
            if 'correction_factor' in data:
                correction_map[y, x] = data['correction_factor']
            elif 'correction_factors' in data and 'total' in data['correction_factors']:
                correction_map[y, x] = data['correction_factors']['total']
    
    # Calculate confidence map
    # Areas with high R² and low RMSE get high confidence
    valid_mask = ~np.isnan(r_squared_map) & ~np.isnan(rmse_map)
    if valid_mask.any():
        # Normalize RMSE for scaling
        rmse_valid = rmse_map[valid_mask]
        if rmse_valid.size > 0 and np.max(rmse_valid) > 0:
            normalized_rmse = rmse_map / np.max(rmse_valid)
            # Combined metric: high R² and low RMSE give high confidence
            confidence_map[valid_mask] = r_squared_map[valid_mask] * (1 - normalized_rmse[valid_mask])
    
    # Calculate precision map (inverse of relative uncertainty)
    # This could be estimated from statistical properties of the fit
    # Higher R² generally correlates with lower uncertainty
    precision_map[valid_mask] = np.sqrt(r_squared_map[valid_mask])
    
    # Collect all maps in a dictionary
    quality_maps = {
        'r_squared': r_squared_map,
        'rmse': rmse_map,
        'confidence': confidence_map,
        'precision': precision_map,
        'inversion_efficiency': inv_efficiency_map,
        'correction_factor': correction_map
    }
    
    return quality_maps

def visualize_quality_maps(quality_maps, fig=None, vmin_percentile=5, vmax_percentile=95):
    """
    Visualize multiple quality maps in a single figure
    
    Parameters:
    -----------
    quality_maps : dict
        Dictionary of quality maps from create_comprehensive_quality_maps
    fig : matplotlib.figure.Figure, optional
        Figure to use for plotting
    vmin_percentile, vmax_percentile : int, optional
        Percentiles to use for colormap scaling
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with quality map visualizations
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create figure if not provided
    if fig is None:
        fig = plt.figure(figsize=(15, 10))
    
    # Number of maps to display
    n_maps = len(quality_maps)
    n_cols = min(3, n_maps)
    n_rows = (n_maps + n_cols - 1) // n_cols
    
    # Colormaps for different metrics
    colormaps = {
        'r_squared': 'viridis',
        'rmse': 'hot_r',  # Reversed hot (lower RMSE is better)
        'confidence': 'plasma',
        'precision': 'cividis',
        'inversion_efficiency': 'cool',
        'correction_factor': 'coolwarm'
    }
    
    # Titles and descriptions
    titles = {
        'r_squared': 'R² (Goodness of Fit)',
        'rmse': 'RMSE (Fitting Error)',
        'confidence': 'Confidence Score',
        'precision': 'Precision Estimate',
        'inversion_efficiency': 'Inversion Efficiency',
        'correction_factor': 'Total Correction Factor'
    }
    
    # Plot each map
    for i, (key, quality_map) in enumerate(quality_maps.items()):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Get valid values for percentile calculation
        valid_values = quality_map[~np.isnan(quality_map)]
        
        if len(valid_values) > 0:
            vmin = np.percentile(valid_values, vmin_percentile)
            vmax = np.percentile(valid_values, vmax_percentile)
            
            # Use custom limits for specific maps
            if key == 'r_squared':
                vmin = max(0, vmin)
                vmax = min(1, vmax)
            elif key == 'confidence':
                vmin = max(0, vmin)
                vmax = min(1, vmax)
            elif key == 'inversion_efficiency':
                vmin = max(0.5, vmin)
                vmax = min(1.0, vmax)
            
            # Get colormap
            cmap = colormaps.get(key, 'viridis')
            
            # Plot with masked NaN values
            masked_map = np.ma.masked_invalid(quality_map)
            im = ax.imshow(masked_map, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Add statistics to title
            stats_str = ""
            if len(valid_values) > 0:
                mean_val = np.mean(valid_values)
                median_val = np.median(valid_values)
                stats_str = f"Mean: {mean_val:.2f}, Median: {median_val:.2f}"
            
            # Set title
            ax.set_title(f"{titles.get(key, key)}\n{stats_str}")
        else:
            # No valid data
            ax.text(0.5, 0.5, "No valid data", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
    fig.tight_layout()
    return fig

def generate_quality_report(t1_map, pixel_cache, sequence_params, roi_coords=None):
    """
    Generate a comprehensive quality report for a T1 mapping session
    
    Parameters:
    -----------
    t1_map : ndarray
        T1 map
    pixel_cache : dict
        Dictionary of pixel-wise fitting results
    sequence_params : dict
        Sequence parameters (TR, flip angle, etc.)
    roi_coords : tuple, optional
        ROI coordinates (x1, x2, y1, y2)
        
    Returns:
    --------
    report : dict
        Dictionary containing quality metrics and statistics
    """
    import numpy as np
    
    # Initialize report
    report = {
        'summary': {},
        'sequence': sequence_params.copy(),
        'statistics': {},
        'quality': {},
        'warnings': []
    }
    
    # Get valid T1 values
    valid_t1 = t1_map[~np.isnan(t1_map)]
    
    # Calculate basic statistics
    if len(valid_t1) > 0:
        report['statistics']['t1_mean'] = float(np.mean(valid_t1))
        report['statistics']['t1_median'] = float(np.median(valid_t1))
        report['statistics']['t1_std'] = float(np.std(valid_t1))
        report['statistics']['t1_min'] = float(np.min(valid_t1))
        report['statistics']['t1_max'] = float(np.max(valid_t1))
        report['statistics']['t1_p05'] = float(np.percentile(valid_t1, 5))
        report['statistics']['t1_p95'] = float(np.percentile(valid_t1, 95))
        report['statistics']['valid_pixel_percentage'] = 100 * len(valid_t1) / t1_map.size
    else:
        report['warnings'].append("No valid T1 values found in map")
    
    # Extract quality metrics from pixel cache
    r_squared_values = []
    rmse_values = []
    model_types = {}
    fallback_count = 0
    
    for key, data in pixel_cache.items():
        if 'r_squared' in data and data['r_squared'] is not None:
            r_squared_values.append(data['r_squared'])
        
        if 'rmse' in data and data['rmse'] is not None:
            rmse_values.append(data['rmse'])
        
        # Count model types used
        if 'model' in data:
            model_type = data['model']
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        # Count fallback uses
        if data.get('fallback_used', False):
            fallback_count += 1
    
    # Store quality metrics
    if r_squared_values:
        report['quality']['mean_r_squared'] = float(np.mean(r_squared_values))
        report['quality']['median_r_squared'] = float(np.median(r_squared_values))
        
        # Define quality thresholds
        if report['quality']['median_r_squared'] < 0.7:
            report['warnings'].append("Low median R-squared value (<0.7), suggesting poor fitting quality")
        
    if rmse_values:
        report['quality']['mean_rmse'] = float(np.mean(rmse_values))
    
    # Report on model types
    report['quality']['model_types'] = model_types
    report['quality']['fallback_percentage'] = 100 * fallback_count / len(pixel_cache) if pixel_cache else 0
    
    if report['quality'].get('fallback_percentage', 0) > 30:
        report['warnings'].append(f"High fallback rate ({report['quality']['fallback_percentage']:.1f}%), suggesting fitting difficulties")
    
    # Check correction parameters
    if sequence_params.get('FlipAngle') is None:
        report['warnings'].append("Missing flip angle data, which may affect correction accuracy")
    
    if sequence_params.get('TR') is None:
        report['warnings'].append("Missing TR data, which may affect correction accuracy")
    
    # Check T1 range for physiological plausibility
    if valid_t1.size > 0:
        if np.median(valid_t1) < 200 or np.median(valid_t1) > 2500:
            report['warnings'].append(f"Median T1 value ({np.median(valid_t1):.1f} ms) outside typical physiological range")
    
    # Overall quality score (0-100)
    quality_score = 0
    factors = []
    
    if r_squared_values:
        # R-squared contribution (0-40 points)
        r_squared_factor = min(1.0, max(0, np.median(r_squared_values)))
        r_squared_score = 40 * r_squared_factor
        factors.append(('R-squared', r_squared_score))
        quality_score += r_squared_score
    
    # Success rate contribution (0-30 points)
    if t1_map.size > 0:
        success_rate = len(valid_t1) / t1_map.size
        success_score = 30 * success_rate
        factors.append(('Success rate', success_score))
        quality_score += success_score
    
    # Fallback rate contribution (0-20 points)
    if pixel_cache:
        fallback_rate = fallback_count / len(pixel_cache)
        fallback_score = 20 * (1 - fallback_rate)
        factors.append(('Fallback penalty', fallback_score))
        quality_score += fallback_score
    
    # Parameter completeness (0-10 points)
    param_score = 10
    if sequence_params.get('FlipAngle') is None:
        param_score -= 5
    if sequence_params.get('TR') is None:
        param_score -= 5
    factors.append(('Parameters', param_score))
    quality_score += param_score
    
    # Store quality score and factors
    report['summary']['quality_score'] = round(quality_score)
    report['summary']['quality_factors'] = factors
    
    # Overall quality assessment
    if quality_score >= 85:
        report['summary']['quality_assessment'] = "Excellent"
    elif quality_score >= 70:
        report['summary']['quality_assessment'] = "Good"
    elif quality_score >= 50:
        report['summary']['quality_assessment'] = "Fair"
    else:
        report['summary']['quality_assessment'] = "Poor"
        
    return report

def enhanced_t1_correction_gre_flexible(t1_apparent, a, b, flip_angle=None, tr=None, te=None, t2star=None, 
                                       inversion_efficiency=None, b1_map=None, 
                                       apply_look_locker=True, apply_t2star=False,
                                       apply_saturation=False, apply_inversion=False,
                                       apply_cardiac_correction=False,
                                       model_type='simple'):
    """Enhanced T1 correction with proper GRE MOLLI physics and cardiac-specific correction"""
    import numpy as np
    
    # Initialize correction factors dictionary
    correction_factors = {
        'look_locker': 1.0,
        'saturation': 1.0,
        't2star_bias': 1.0,  # Changed from 'readout' to be more descriptive
        'inversion': 1.0,
        'cardiac': 1.0,
        'total': 1.0
    }
    
    # Type checking
    is_array = isinstance(t1_apparent, np.ndarray)
    
    # Function to ensure values are within physical limits
    def constrain_t1(t1):
        if is_array:
            return np.clip(t1, 50, 5000)
        else:
            return max(50, min(5000, t1))
    
    # Apply initial constraint and create working copy
    t1_apparent = constrain_t1(t1_apparent)
    
    # Create explicit working copy to avoid view issues
    if is_array:
        t1_corrected = t1_apparent.copy()
    else:
        t1_corrected = float(t1_apparent)
    
    # 1. Look-Locker correction (most important for MOLLI)
    if apply_look_locker:
        if flip_angle is not None and tr is not None:
            # Use the new proper Look-Locker correction function
            look_locker_factor = calculate_look_locker_correction_factor(
                t1_apparent, flip_angle, tr, model_type='MOLLI'
            )
            
            # Handle array/scalar cases
            if is_array and not isinstance(look_locker_factor, np.ndarray):
                look_locker_factor = np.full_like(t1_apparent, look_locker_factor)
        else:
            # Fallback to model-specific empirical values
            if model_type.lower() == 'classic':
                if isinstance(b, (int, float)) and 0.5 < b < 1.0:
                    look_locker_factor = 1.0 / (2 * b - 1)
                    look_locker_factor = max(1.1, min(1.5, look_locker_factor))
                else:
                    look_locker_factor = 1.4
            else:
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0:
                    look_locker_factor = b / a
                    look_locker_factor = max(1.1, min(1.8, look_locker_factor))
                else:
                    look_locker_factor = 1.4
        
        # Apply correction
        t1_corrected = t1_corrected * look_locker_factor
        correction_factors['look_locker'] = look_locker_factor if not is_array else np.mean(look_locker_factor)
    
    # 2. T2* bias correction (NOT T2* decay correction)
    if apply_t2star and t2star is not None and te is not None:
        if te > 0 and t2star > 0:
            # Use the proper T2* bias estimation function
            t2star_bias_factor = estimate_t2star_bias_in_molli(te, t2star)
            
            # Apply the bias correction
            t1_corrected = t1_corrected * t2star_bias_factor
            correction_factors['t2star_bias'] = t2star_bias_factor
    
    # 3. Saturation correction - compensates for incomplete recovery
    if apply_saturation and tr is not None:
        sat_factor = 1.0
        
        if tr > 0:
            # Handle both array and scalar cases properly
            if is_array:
                valid_mask = t1_corrected > 0
                tr_t1_ratio = np.zeros_like(t1_corrected)
                tr_t1_ratio[valid_mask] = tr / t1_corrected[valid_mask]
                
                # Simplified correction model based on steady-state GRE
                sat_factor = np.ones_like(t1_corrected)
                exp_term = np.exp(-tr_t1_ratio)
                exp_term = np.clip(exp_term, 0, 0.999)  # Avoid division by zero
                sat_factor[valid_mask] = 1.0 / (1.0 - exp_term[valid_mask])
                sat_factor = np.clip(sat_factor, 1.0, 1.5)
            else:
                if t1_corrected > 0:
                    tr_t1_ratio = tr / t1_corrected
                    exp_term = np.exp(-tr_t1_ratio)
                    if exp_term < 0.999:  # Avoid division by zero
                        sat_factor = 1.0 / (1.0 - exp_term)
                        sat_factor = max(1.0, min(1.5, sat_factor))
        
        # Apply correction
        t1_corrected = t1_corrected * sat_factor
        correction_factors['saturation'] = sat_factor if not is_array else np.mean(sat_factor)
    
    # 4. Inversion efficiency correction
    if apply_inversion and inversion_efficiency is not None:
        inv_factor = 1.0
        
        if 0 < inversion_efficiency < 1.0:
            # Calculate correction based on inversion efficiency
            inv_factor = 1.0 / inversion_efficiency
            
            # Apply reasonable bounds
            if is_array:
                if isinstance(inv_factor, np.ndarray):
                    inv_factor = np.clip(inv_factor, 1.0, 2.0)
                else:
                    inv_factor = max(1.0, min(2.0, inv_factor))
            else:
                inv_factor = max(1.0, min(2.0, inv_factor))
        
        # Apply correction
        t1_corrected = t1_corrected * inv_factor
        correction_factors['inversion'] = inv_factor
    
    # 5. Cardiac-specific correction if requested
    if apply_cardiac_correction:
        # Literature-based empirical correction for cardiac tissue
        cardiac_factor = 1.12  # Additional 12% increase
        
        # Apply with proper array handling
        if is_array:
            valid_mask = ~np.isnan(t1_corrected)
            if valid_mask.any():
                # Create explicit copy of values to modify
                temp_values = t1_corrected[valid_mask].copy()
                temp_values = temp_values * cardiac_factor
                t1_corrected[valid_mask] = temp_values
        else:
            if not np.isnan(t1_corrected):
                t1_corrected = t1_corrected * cardiac_factor
        
        correction_factors['cardiac'] = cardiac_factor
    
    # Calculate total correction factor
    correction_factors['total'] = (correction_factors['look_locker'] * 
                                  correction_factors['saturation'] * 
                                  correction_factors['t2star_bias'] * 
                                  correction_factors['inversion'] *
                                  correction_factors['cardiac'])
    
    # Return final constrained T1 with explicit copy
    final_t1 = constrain_t1(t1_corrected)
    if is_array:
        final_t1 = final_t1.copy()  # Ensure we return a copy, not a view
    
    return final_t1, correction_factors

def enhanced_t1_correction_gre(t1_apparent, a, b, flip_angle=None, tr=None, te=None, 
                              t2star=None, inversion_efficiency=None, 
                              correction_method='look_locker'):
    """
    Wrapper function for standard correction methods
    Calls the flexible function with appropriate flags
    """
    # Map correction methods to flags
    if correction_method == 'none':
        apply_ll = False
        apply_t2star = False
        apply_sat = False
        apply_inv = False
    elif correction_method == 'look_locker':
        apply_ll = True
        apply_t2star = False
        apply_sat = False
        apply_inv = False
    elif correction_method == 'full':
        apply_ll = True
        apply_t2star = True
        apply_sat = True
        apply_inv = True
    elif correction_method == 'look_locker_t2star':
        apply_ll = True
        apply_t2star = True
        apply_sat = False
        apply_inv = False
    elif correction_method == 'look_locker_inversion':
        apply_ll = True
        apply_t2star = False
        apply_sat = False
        apply_inv = True
    elif correction_method == 'standard_gre':
        apply_ll = True
        apply_t2star = False
        apply_sat = True
        apply_inv = False
    elif correction_method.startswith('custom:'):
        # Extract which corrections to apply from the string
        correction_codes = correction_method[7:]  # Remove 'custom:' prefix
        apply_ll = 'L' in correction_codes
        apply_t2star = 'T' in correction_codes
        apply_sat = 'S' in correction_codes
        apply_inv = 'B' in correction_codes
    else:
        # Default to Look-Locker only
        apply_ll = True
        apply_t2star = False
        apply_sat = False
        apply_inv = False
    
    return enhanced_t1_correction_gre_flexible(
        t1_apparent, a, b,
        flip_angle, tr, te, t2star,
        inversion_efficiency,
        b1_map=None,
        apply_look_locker=apply_ll,
        apply_t2star=apply_t2star,
        apply_saturation=apply_sat,
        apply_inversion=apply_inv,
        apply_cardiac_correction=False,
        model_type='simple'
    )


def apply_demons_motion_correction(image_stack):
    """
    Apply demons motion correction to an image stack
    
    Parameters:
    -----------
    image_stack : ndarray
        Stack of images with shape [n_timepoints, height, width]
    
    Returns:
    --------
    corrected_stack : ndarray
        Motion corrected image stack
    """
    corrected_stack = []
    sitk_images = [sitk.GetImageFromArray(image_stack[i]) for i in range(image_stack.shape[0])]
    reference = sitk_images[0]

    for i, moving in enumerate(sitk_images):
        # Use demons registration (non-rigid)
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(50)
        demons.SetStandardDeviations(2.0)
        
        # Initial alignment with rigid transform
        initial_transform = sitk.Euler2DTransform()
        initializer = sitk.CenteredTransformInitializer(reference, moving, initial_transform, 
                                                       sitk.CenteredTransformInitializerFilter.GEOMETRY)
        moving_initial = sitk.Resample(moving, reference, initializer, sitk.sitkLinear, 0.0, moving.GetPixelID())
        
        # Run demons on pre-aligned images
        displacement_field = demons.Execute(reference, moving_initial)
        
        # Create a displacement field transform from the displacement field
        displacement_transform = sitk.DisplacementFieldTransform(displacement_field)
        
        # Apply the displacement field transform
        resampled = sitk.Resample(moving, reference, displacement_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
        corrected_stack.append(sitk.GetArrayFromImage(resampled))

    return np.stack(corrected_stack, axis=0)

def enhanced_roi_selection(image, title="Select ROI"):
    """Improved ROI selection with zoom functionality and clearer visualization"""
    coords = {'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0}
    selection_made = [False]  # Use list for nonlocal modification
    
    # Create a figure with proper sizing
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Normalize image for display
    display_img = image.copy()
    if display_img.max() > display_img.min():
        display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min())
    
    # Display the image
    ax.imshow(display_img, cmap='gray')
    ax.set_title(title)
    
    # Add instructions
    ax.text(0.5, 0.01, 
           "Click and drag to select ROI. Use mouse wheel to zoom.\nPress Enter to confirm selection.", 
           transform=fig.transFigure, ha='center', 
           bbox=dict(facecolor='white', alpha=0.8))
    
    # Rectangle selection
    def onselect(eclick, erelease):
        try:
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            # Ensure coordinates are within image bounds
            h, w = image.shape
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w-1))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h-1))
            
            # Store in ascending order
            coords['x1'], coords['y1'] = min(x1, x2), min(y1, y2)
            coords['x2'], coords['y2'] = max(x1, x2), max(y1, y2)
            
            # Update selection rectangle if it exists
            if hasattr(ax, 'roi_rect') and ax.roi_rect:
                ax.roi_rect.set_xy((coords['x1'], coords['y1']))
                ax.roi_rect.set_width(coords['x2'] - coords['x1'])
                ax.roi_rect.set_height(coords['y2'] - coords['y1'])
            else:
                ax.roi_rect = plt.Rectangle(
                    (coords['x1'], coords['y1']), 
                    coords['x2'] - coords['x1'], 
                    coords['y2'] - coords['y1'],
                    edgecolor='red', facecolor='none', linewidth=2
                )
                ax.add_patch(ax.roi_rect)
            
            # Draw dimensions on the rectangle
            if hasattr(ax, 'roi_dims') and ax.roi_dims:
                ax.roi_dims.remove()
            
            # Add dimensions text
            ax.roi_dims = ax.text(
                coords['x2'] + 5, coords['y1'] + 5,
                f"Size: {coords['x2'] - coords['x1']} × {coords['y2'] - coords['y1']} px",
                color='red', backgroundcolor='white'
            )
            
            fig.canvas.draw()
            selection_made[0] = True
        except (TypeError, ValueError) as e:
            print(f"ROI selection error: {e}")
    
    # Add zoom functionality with mouse wheel
    def zoom_factory(ax, base_scale=1.1):
        """Implements zooming with the mouse wheel"""
        def zoom(event):
            if event.inaxes != ax:
                return
            
            # Get the current x and y limits
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            
            # Get event location
            xdata, ydata = event.xdata, event.ydata
            if xdata is None or ydata is None:
                return
                
            # Determine zoom factor
            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                scale_factor = 1
            
            # Calculate new limits
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            # Set new limits
            ax.set_xlim([xdata - new_width * (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]),
                         xdata + new_width * (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])])
            ax.set_ylim([ydata - new_height * (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0]),
                         ydata + new_height * (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])])
            
            fig.canvas.draw()
        
        # Connect scroll event
        fig.canvas.mpl_connect('scroll_event', zoom)
        return zoom
    
    # Connect the zoom function
    zoom_factory(ax)
    
    # Handle keyboard events for confirmation
    def on_key(event):
        if event.key == 'enter':
            # Validate the selection
            if selection_made[0] and coords['x2'] > coords['x1'] and coords['y2'] > coords['y1']:
                plt.close(fig)
            else:
                # Alert the user if no valid selection
                ax.set_title("Please make a valid selection first!")
                fig.canvas.draw()
    
    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Create and connect the selector
    rect_selector = RectangleSelector(
        ax, onselect, interactive=True, 
        minspanx=5, minspany=5, 
        spancoords='pixels',
        button=[1]  # Left mouse button only
    )
    
    plt.tight_layout()
    plt.show()
    
    # Return the selected coordinates
    if coords['x2'] <= coords['x1'] or coords['y2'] <= coords['y1']:
        # If selection wasn't completed properly, use center region
        h, w = image.shape
        coords['x1'], coords['y1'] = w//4, h//4
        coords['x2'], coords['y2'] = 3*w//4, 3*h//4
        print("Invalid selection, using default ROI")
    
    return coords['x1'], coords['x2'], coords['y1'], coords['y2']

def get_colormap_list():
    """Return a list of available colormaps"""
    colormaps = [
        'turbo', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'jet', 'rainbow', 'hsv', 'hot', 'cool', 'coolwarm'
    ]
    return colormaps

def apply_colormap(t1_map, colormap_name='turbo', min_val=None, max_val=None, alpha=0.7):
    """Apply a colormap to a T1 map with transparency"""
    cmap = plt.get_cmap(colormap_name)
    
    # Set default min/max if not provided
    if min_val is None:
        min_val = np.nanmin(t1_map)
    if max_val is None:
        max_val = np.nanmax(t1_map)
    
    # Check for invalid values or zero range
    if np.isnan(min_val) or np.isnan(max_val) or min_val >= max_val:
        # Return a fully transparent image if values are invalid
        rgba_img = np.zeros((*t1_map.shape, 4))
        print(f"Warning: Invalid T1 range ({min_val} to {max_val}), showing transparent overlay")
        return rgba_img
    
    # Normalize the T1 values to the range [0, 1]
    norm_t1 = np.clip((t1_map - min_val) / (max_val - min_val), 0, 1)
    
    # Create RGBA image using the colormap
    rgba_img = cmap(norm_t1)
    
    # Set alpha (transparency) value for non-NaN pixels
    rgba_img[..., 3] = np.where(np.isnan(t1_map), 0, alpha)
    
    return rgba_img

def calculate_default_window_level(image):
    """Calculate reasonable default window/level values for an image"""
    # For MRI images, using percentiles often works better than min/max
    p02 = np.percentile(image, 2)
    p98 = np.percentile(image, 98)
    
    window = p98 - p02
    level = (p98 + p02) / 2
    
    return window, level

def apply_window_level(image, window, level):
    """Apply window/level (width/center) to an image for display"""
    vmin = level - window/2
    vmax = level + window/2
    display_img = np.clip(image, vmin, vmax)
    display_img = (display_img - vmin) / (vmax - vmin)
    return display_img

def coregister_b1_to_reference(b1_map, reference_image, method='affine', visualize=False):
    """
    Coregister a B1 map to match the dimensions and FOV of a reference image
    
    Parameters:
    -----------
    b1_map : ndarray
        B1 map to be registered
    reference_image : ndarray
        Reference image (e.g., T1 map or original MR image)
    method : str, optional
        Registration method ('rigid', 'affine', 'bspline', or 'demons')
    visualize : bool, optional
        Whether to visualize registration result
        
    Returns:
    --------
    registered_b1 : ndarray
        B1 map registered to match reference image dimensions
    """
    # Convert arrays to SimpleITK images
    b1_sitk = sitk.GetImageFromArray(b1_map.astype(np.float32))
    ref_sitk = sitk.GetImageFromArray(reference_image.astype(np.float32))
    
    # Normalize images for better registration
    b1_norm = sitk.Normalize(b1_sitk)
    ref_norm = sitk.Normalize(ref_sitk)
    
    # Set up registration method
    if method == 'rigid':
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsCorrelation()
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=0.1, minStep=1e-4, numberOfIterations=100)
        registration_method.SetInitialTransform(
            sitk.CenteredTransformInitializer(
                ref_norm, b1_norm, sitk.Euler2DTransform(), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY))
        
        # Execute registration
        transform = registration_method.Execute(ref_norm, b1_norm)
        
    elif method == 'affine':
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsCorrelation()
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=0.1, minStep=1e-4, numberOfIterations=100)
        registration_method.SetInitialTransform(
            sitk.CenteredTransformInitializer(
                ref_norm, b1_norm, sitk.AffineTransform(2), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY))
        
        # Execute registration
        transform = registration_method.Execute(ref_norm, b1_norm)
        
    elif method == 'bspline':
        # More advanced B-spline registration
        transformDomainMeshSize = [8, 8]
        
        # Set up initial transform
        initial_transform = sitk.CenteredTransformInitializer(
            ref_sitk, b1_sitk, sitk.AffineTransform(2),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        
        # Apply initial transform
        b1_resampled = sitk.Resample(
            b1_sitk, ref_sitk, initial_transform, 
            sitk.sitkLinear, 0.0, b1_sitk.GetPixelID())
        
        # Set up B-spline transform
        transform = sitk.BSplineTransformInitializer(
            ref_sitk, transformDomainMeshSize)
        
        # Set up registration method
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsCorrelation()
        registration_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=100)
        registration_method.SetInitialTransform(transform)
        
        # Execute registration
        transform = registration_method.Execute(ref_sitk, b1_resampled)
        
    elif method == 'demons':
        # Demons registration
        # Set up initial transform for pre-alignment
        initial_transform = sitk.CenteredTransformInitializer(
            ref_sitk, b1_sitk, sitk.AffineTransform(2),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        
        # Pre-align with rigid transform
        b1_resampled = sitk.Resample(
            b1_sitk, ref_sitk, initial_transform, 
            sitk.sitkLinear, 0.0, b1_sitk.GetPixelID())
        
        # Set up demons registration
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(50)
        demons.SetStandardDeviations(2.0)
        
        # Execute demons registration
        displacement_field = demons.Execute(ref_norm, sitk.Normalize(b1_resampled))
        
        # Create transform from displacement field
        transform = sitk.DisplacementFieldTransform(displacement_field)
    
    else:
        raise ValueError(f"Unknown registration method: {method}")
    
    # Resample the B1 map to match reference dimensions
    resampled_b1 = sitk.Resample(
        b1_sitk, ref_sitk, transform,
        sitk.sitkLinear, 0.0, b1_sitk.GetPixelID())
    
    # Convert back to numpy array
    registered_b1 = sitk.GetArrayFromImage(resampled_b1)
    
    # Visualize results if requested
    if visualize:
        # Create a figure to display registration results
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Original images
        axes[0, 0].imshow(b1_map, cmap='viridis')
        axes[0, 0].set_title('Original B1 Map')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(reference_image, cmap='gray')
        axes[0, 1].set_title('Reference Image')
        axes[0, 1].axis('off')
        
        # Registered result
        axes[1, 0].imshow(registered_b1, cmap='viridis')
        axes[1, 0].set_title('Registered B1 Map')
        axes[1, 0].axis('off')
        
        # Overlay to check alignment
        overlay = np.zeros((*registered_b1.shape, 3))
        # Red channel: registered B1 map (normalized)
        b1_min, b1_max = registered_b1.min(), registered_b1.max()
        if b1_max > b1_min:
            overlay[..., 0] = (registered_b1 - b1_min) / (b1_max - b1_min)
        
        # Green channel: reference image (normalized)
        ref_min, ref_max = reference_image.min(), reference_image.max()
        if ref_max > ref_min:
            overlay[..., 1] = (reference_image - ref_min) / (ref_max - ref_min)
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (R: B1, G: Reference)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return registered_b1

def validate_t1_results(t1_map, expected_range=(200, 3000)):
    """
    Validate T1 mapping results for physically reasonable values
    """
    if t1_map is None:
        return False, "T1 map is None"
    
    valid_t1s = t1_map[~np.isnan(t1_map)]
    
    if len(valid_t1s) == 0:
        return False, "No valid T1 values found"
    
    # Check for reasonable T1 range
    median_t1 = np.median(valid_t1s)
    if median_t1 < expected_range[0] or median_t1 > expected_range[1]:
        return False, f"Median T1 ({median_t1:.1f} ms) outside expected range {expected_range}"
    
    # Check for excessive variation
    cv = np.std(valid_t1s) / np.mean(valid_t1s)
    if cv > 0.5:  # Coefficient of variation > 50%
        return False, f"Excessive T1 variation (CV = {cv:.2f})"
    
    # Check success rate
    success_rate = len(valid_t1s) / t1_map.size
    if success_rate < 0.5:  # Less than 50% successful fits
        return False, f"Low success rate ({success_rate:.1%})"
    
    return True, "T1 results appear valid"

def save_t1_map_as_dicom(t1_map, output_path, reference_dicom_path=None, 
                        sequence_params=None, roi_coords=None):
    """
    Save T1 map as a DICOM file with proper headers
    
    Parameters:
    -----------
    t1_map : ndarray
        T1 map in milliseconds
    output_path : str
        Output DICOM file path
    reference_dicom_path : str, optional
        Path to reference DICOM for copying headers
    sequence_params : dict, optional
        Sequence parameters
    roi_coords : tuple, optional
        ROI coordinates if applicable
    """
    
    # Load reference DICOM if provided
    if reference_dicom_path and os.path.exists(reference_dicom_path):
        try:
            ref_ds = pydicom.dcmread(reference_dicom_path)
        except:
            ref_ds = None
    else:
        ref_ds = None
    
    # Create new DICOM dataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
    
    # Create file dataset
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Copy basic patient/study info from reference if available
    if ref_ds:
        # Patient information
        for tag in ['PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex']:
            if hasattr(ref_ds, tag):
                setattr(ds, tag, getattr(ref_ds, tag))
        
        # Study information
        for tag in ['StudyInstanceUID', 'StudyDate', 'StudyTime', 'StudyID', 'StudyDescription']:
            if hasattr(ref_ds, tag):
                setattr(ds, tag, getattr(ref_ds, tag))
        
        # Series information (create new series for T1 map)
        ds.SeriesInstanceUID = generate_uid()
        if hasattr(ref_ds, 'SeriesNumber'):
            ds.SeriesNumber = str(int(getattr(ref_ds, 'SeriesNumber', '1')) + 1000)
        else:
            ds.SeriesNumber = "1001"
        
        # Copy spatial information
        for tag in ['ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing', 
                   'SliceThickness', 'SliceLocation']:
            if hasattr(ref_ds, tag):
                setattr(ds, tag, getattr(ref_ds, tag))
    else:
        # Default values if no reference
        ds.PatientName = "T1_MAPPING_PATIENT"
        ds.PatientID = "T1MAP001"
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = "1001"
        ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
    
    # T1 map specific information
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = 'MR'
    ds.SeriesDescription = 'T1_MAP_GRE_MOLLI'
    ds.ImageType = ['DERIVED', 'PRIMARY', 'T1_MAP']
    
    # Sequence information
    if sequence_params:
        if 'TR' in sequence_params and sequence_params['TR']:
            ds.RepetitionTime = float(sequence_params['TR'])
        if 'TE' in sequence_params and sequence_params['TE']:
            ds.EchoTime = float(sequence_params['TE'])
        if 'FlipAngle' in sequence_params and sequence_params['FlipAngle']:
            ds.FlipAngle = float(sequence_params['FlipAngle'])
    
    # Convert T1 map to proper format
    # T1 values should be in milliseconds, convert to uint16
    t1_scaled = np.nan_to_num(t1_map, nan=0, posinf=4095, neginf=0)
    t1_scaled = np.clip(t1_scaled, 0, 4095)  # Fit in 12-bit range
    t1_uint16 = t1_scaled.astype(np.uint16)
    
    # Image data
    ds.Rows, ds.Columns = t1_uint16.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    
    # Rescale parameters to store T1 values correctly
    ds.RescaleSlope = 1.0  # T1 values are direct
    ds.RescaleIntercept = 0.0
    ds.RescaleType = 'T1'  # Indicates T1 mapping
    
    # Add T1 mapping specific tags
    ds.add_new([0x0018, 0x9301], 'US', 1)  # Partial Fourier
    ds.add_new([0x0018, 0x9302], 'CS', 'T1')  # MR Acquisition Type
    
    # Add ROI information if available
    if roi_coords:
        x1, x2, y1, y2 = roi_coords
        roi_description = f"ROI: x={x1}-{x2}, y={y1}-{y2}"
        ds.add_new([0x0008, 0x1040], 'LO', roi_description)  # Institutional Department Name
    
    # Instance creation date/time
    now = datetime.datetime.now()
    ds.InstanceCreationDate = now.strftime('%Y%m%d')
    ds.InstanceCreationTime = now.strftime('%H%M%S')
    ds.ContentDate = now.strftime('%Y%m%d')
    ds.ContentTime = now.strftime('%H%M%S')
    
    # Set pixel data
    ds.PixelData = t1_uint16.tobytes()
    
    # Window/Level for display
    valid_t1 = t1_scaled[t1_scaled > 0]
    if len(valid_t1) > 0:
        ds.WindowCenter = str(int(np.median(valid_t1)))
        ds.WindowWidth = str(int(np.percentile(valid_t1, 95) - np.percentile(valid_t1, 5)))
    else:
        ds.WindowCenter = "1000"
        ds.WindowWidth = "1000"
    
    # Save the DICOM file
    ds.save_as(output_path, write_like_original=False)
    print(f"T1 map saved as DICOM: {output_path}")

class LeanT1MappingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced GRE MOLLI T1 Mapping Tool")
        self.root.geometry("1200x700")
        
        # Set theme
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            pass  # Fallback to default theme if 'clam' not available
        
        # Configure styles
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TLabel', font=('Arial', 10))
        
        # Instance variables
        self.dicom_files = []
        self.image_stack = None
        self.tis = None
        self.sequence_params = {}
        self.corrected_stack = None
        self.t1_result = None
        self.roi_coords = None
        self.processed_signal = None
        self.fitted_curve = None
        self.t1_map = None
        self.colormap_name = 'turbo'
        self.overlay_alpha = 0.7
        self.show_overlay = True
        self.show_corrected_fit = False  # Add this line - default to only showing the apparent fit
        self.t1_min = None  # Auto-determined
        self.t1_max = None  # Auto-determined
        self.pixel_cache = {}  # Cache for pixel calculations
        self.b1_map = None  # B1 map data
        self.show_mt_button = None  # Will be created later in UI
        self._custom_correction_string = 'custom:L'  # Default custom correction
        self.t1_map_before_mt = None  # For storing original T1 map
        
        # Add status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                  relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create main frames
        self.create_main_frames()
        
        # Create menu
        self.create_menu()

        # Add DICOM save functionality
        self.add_dicom_save_to_app()

    def create_b1_map_from_inversion_angles(self):
        """GUI method to create B1 map from two inversion flip angle series"""
        # [existing code until the process_b1_map function]
        
        def process_b1_map():
            if not self.fa1_files or not self.fa2_files:
                messagebox.showwarning("Missing Files", 
                                     "Please select files for both inversion flip angle series")
                return
            
            try:
                self.update_status("Loading inversion flip angle images...")
                
                # Add debug info
                print(f"Processing {len(self.fa1_files)} FA1 files and {len(self.fa2_files)} FA2 files")
                
                # Get actual angles from input fields
                try:
                    angle1 = float(fa1_var.get())
                    angle2 = float(fa2_var.get())
                    
                    if angle1 <= 0 or angle2 <= 0 or angle1 >= 180 or angle2 >= 180:
                        raise ValueError("Invalid flip angles")
                except:
                    messagebox.showwarning("Invalid Angles", 
                                         "Please enter valid flip angles (0-180°)")
                    return
                
                # Call the standalone functions correctly (without self)
                image_fa1, image_fa2, _, _, inversion_times = load_dual_inversion_angle_images(
                    self.fa1_files, self.fa2_files
                )
                
                self.update_status(f"Calculating B1 map from inversion flip angles {angle1:.1f}° and {angle2:.1f}°...")
                
                # Call the standalone function correctly (without self)
                self.b1_map, confidence_map = calculate_b1_map_from_inversion_angles(
                    image_fa1, image_fa2, angle1, angle2, inversion_times
                )
                
                # Check if we got valid results
                if np.sum(self.b1_map != 1.0) < 100:  # Very few non-default values
                    messagebox.showwarning("B1 Calculation Warning", 
                                         "Few valid B1 values were calculated. Results may not be reliable.")
                
                # Update UI even if results are sparse
                self.b1_info_var.set(f"B1 map calculated from inversion flip angles\n"
                                   f"FA1: {angle1:.1f}°, FA2: {angle2:.1f}°\n"
                                   f"Size: {self.b1_map.shape[0]}×{self.b1_map.shape[1]} px\n"
                                   f"Range: {np.min(self.b1_map):.2f}-{np.max(self.b1_map):.2f}")
                
                # Enable B1 correction button if T1 map exists
                if hasattr(self, 't1_map') and self.t1_map is not None:
                    self.apply_b1_button.configure(state=tk.NORMAL)
                
                # Enable coregistration button
                if hasattr(self, 'coregister_b1_button'):
                    self.coregister_b1_button.configure(state=tk.NORMAL)
                
                # Enable save B1 map button
                if hasattr(self, 'save_b1_button'):
                    self.save_b1_button.configure(state=tk.NORMAL)
                
                # Show B1 map visualization
                self.show_b1_calculation_results(image_fa1, image_fa2, 
                                               self.b1_map, confidence_map,
                                               angle1, angle2)
                
                self.update_status("B1 map calculated successfully")
                dialog.destroy()
                
            except Exception as e:
                import traceback
                traceback.print_exc()  # Print full traceback for debugging
                self.update_status(f"Error calculating B1 map: {str(e)}")
                show_error_dialog("B1 Calculation Error", 
                                f"Error calculating B1 map from inversion flip angles:\n{str(e)}", e)
        
        ttk.Button(button_frame, text="Calculate B1 Map", 
                  command=process_b1_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def show_b1_calculation_results(self, image_fa1, image_fa2, b1_map, confidence_map, fa1, fa2):
        """
        Show the results of B1 map calculation with better error handling
        """
        # Create a new window for results
        result_window = tk.Toplevel(self.root)
        result_window.title("B1 Map Calculation Results")
        result_window.geometry("1200x800")
        
        # Create figure
        fig = plt.Figure(figsize=(12, 8))
        
        # First flip angle image
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(image_fa1, cmap='gray')
        ax1.set_title(f"Inversion FA {fa1:.1f}°")
        ax1.axis('off')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Second flip angle image
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(image_fa2, cmap='gray')
        ax2.set_title(f"Inversion FA {fa2:.1f}°")
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Signal ratio with robust handling
        ax3 = fig.add_subplot(2, 3, 3)
        ratio_map = np.zeros_like(image_fa1)
        
        # Use safe computation
        safe_mask = (image_fa2 > np.max(image_fa2) * 0.05)
        if np.any(safe_mask):
            safe_image_fa2 = np.maximum(image_fa2, 1e-6)  # Avoid division by zero
            ratio_map[safe_mask] = image_fa1[safe_mask] / safe_image_fa2[safe_mask]
        
        # Limit ratio range for better visualization
        ratio_map_clipped = np.clip(ratio_map, 0.5, 1.5)
        
        im3 = ax3.imshow(ratio_map_clipped, cmap='coolwarm', vmin=0.5, vmax=1.5)
        ax3.set_title("Signal Ratio (FA1/FA2)")
        ax3.axis('off')
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # B1 map
        ax4 = fig.add_subplot(2, 3, 4)
        im4 = ax4.imshow(b1_map, cmap='viridis', vmin=0.7, vmax=1.3)
        ax4.set_title("B1 Map")
        ax4.axis('off')
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('B1 Factor')
        
        # Confidence map
        ax5 = fig.add_subplot(2, 3, 5)
        im5 = ax5.imshow(confidence_map, cmap='plasma', vmin=0, vmax=1)
        ax5.set_title("Confidence Map")
        ax5.axis('off')
        cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.set_label('Confidence')
        
        # Histogram of B1 values - WITH BETTER ERROR HANDLING
        ax6 = fig.add_subplot(2, 3, 6)
        
        # Use a higher confidence threshold to get meaningful values
        valid_b1 = b1_map[(confidence_map > 0.3) & (b1_map >= 0.5) & (b1_map <= 1.5)]
        
        # Check if we have any valid B1 values before making histogram
        if len(valid_b1) > 10:  # Need enough points for a meaningful histogram
            ax6.hist(valid_b1.flatten(), bins=50, density=True, alpha=0.7, color='blue')
            ax6.axvline(1.0, color='red', linestyle='--', label='Nominal B1')
        else:
            ax6.text(0.5, 0.5, "Insufficient high-confidence B1 values", 
                    ha='center', va='center', transform=ax6.transAxes,
                    color='red', fontsize=12)
        
        ax6.set_xlabel('B1 Factor')
        ax6.set_ylabel('Frequency')
        ax6.set_title('B1 Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add statistics text - WITH ROBUST ERROR HANDLING
        stats_text = f"B1 Statistics:\n"
        
        if len(valid_b1) > 10:
            # We have enough valid data points
            stats_text += f"Mean: {np.mean(valid_b1):.3f}\n"
            stats_text += f"Median: {np.median(valid_b1):.3f}\n"
            stats_text += f"Std Dev: {np.std(valid_b1):.3f}\n"
            stats_text += f"Range: [{np.min(valid_b1):.3f}, {np.max(valid_b1):.3f}]\n"
            stats_text += f"Valid pixels: {len(valid_b1)}"
        else:
            # Not enough valid data
            stats_text += "Insufficient high-confidence B1 values\n"
            stats_text += "Try different inversion flip angles\n"
            stats_text += "or check input images"
        
        ax6.text(0.98, 0.98, stats_text, transform=ax6.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        
        # Add canvas to window
        canvas = FigureCanvasTkAgg(fig, result_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add buttons
        button_frame = ttk.Frame(result_window)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_b1_results():
            file_path = filedialog.asksaveasfilename(
                title="Save B1 Map Results",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"B1 map results saved to {file_path}")
        
        ttk.Button(button_frame, text="Save Figure", 
                  command=save_b1_results).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Close", 
                  command=result_window.destroy).pack(side=tk.RIGHT, padx=5)



    def add_dicom_save_to_app(self):
        """
        Add DICOM save functionality to the main application class
        """
        # Add DICOM save button to export section if it exists
        if hasattr(self, 'save_data_button'):
            # Add DICOM save button after existing save buttons
            self.save_dicom_button = ttk.Button(
                self.save_data_button.master, 
                text="Save T1 Map as DICOM", 
                command=self.save_t1_map_dicom,
                state=tk.DISABLED
            )
            self.save_dicom_button.pack(fill=tk.X, pady=5)

    def save_t1_map_dicom(self):
        """Save T1 map as DICOM file"""
        if self.t1_map is None:
            messagebox.showinfo("Info", "Calculate T1 map first")
            return
        
        try:
            # Validate T1 results first
            is_valid, message = validate_t1_results(self.t1_map)
            if not is_valid:
                response = messagebox.askyesno(
                    "T1 Validation Warning", 
                    f"T1 results validation failed: {message}\n\n"
                    "Do you want to save anyway?"
                )
                if not response:
                    return
            
            # Ask for file path
            file_path = filedialog.asksaveasfilename(
                title="Save T1 Map as DICOM",
                defaultextension=".dcm",
                filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            self.update_status("Saving T1 map as DICOM...")
            
            # Get reference DICOM if available
            reference_dicom = None
            if hasattr(self, 'dicom_files') and self.dicom_files:
                reference_dicom = self.dicom_files[0]
            
            # Save T1 map as DICOM
            save_t1_map_as_dicom(
                self.t1_map,
                file_path,
                reference_dicom_path=reference_dicom,
                sequence_params=self.sequence_params,
                roi_coords=self.roi_coords
            )
            
            self.update_status(f"T1 map saved as DICOM: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"T1 map saved as DICOM file:\n{file_path}")
            
        except Exception as e:
            self.update_status("Error saving T1 map as DICOM")
            show_error_dialog("DICOM Save Error", "Error saving T1 map as DICOM", e)

    def create_main_frames(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        self.left_panel = ttk.LabelFrame(self.main_frame, text="Controls", padding=10, width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_panel.pack_propagate(False)  # Prevent shrinking
        
        # Center panel for visualization
        self.center_panel = ttk.LabelFrame(self.main_frame, text="Visualization", padding=10)
        self.center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # NEW: Right panel for additional controls
        self.right_panel = ttk.LabelFrame(self.main_frame, text="Advanced Controls", padding=10, width=300)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_panel.pack_propagate(False)  # Prevent shrinking
        
        # Add content to left panel
        self.create_left_panel_content()
        
        # Add content to right panel
        self.create_right_panel_content()
        
        # Create placeholder for visualization
        self.create_visualization_area()

    def create_left_panel_content(self):
        # Load files section
        load_frame = ttk.LabelFrame(self.left_panel, text="Data Loading", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(load_frame, text="Load DICOM Files", 
                  command=self.load_dicom_files).pack(fill=tk.X, pady=5)
        
        self.file_info_var = tk.StringVar(value="No files loaded")
        ttk.Label(load_frame, textvariable=self.file_info_var, 
                 wraplength=230).pack(fill=tk.X)
        
        # Fitting Options frame
        fitting_frame = ttk.LabelFrame(self.left_panel, text="Fitting Options", padding=10)
        fitting_frame.pack(fill=tk.X, pady=(0, 10))

        # Model selection
        ttk.Label(fitting_frame, text="IR Model:").grid(row=0, column=0, sticky='w', pady=5)
        self.model_var = tk.StringVar(value="simple")
        model_combo = ttk.Combobox(fitting_frame, textvariable=self.model_var, 
                                  values=["simple", "classic"], width=15)
        model_combo.grid(row=0, column=1, padx=5, pady=5)
        model_combo.bind("<<ComboboxSelected>>", self.auto_recalculate_t1)

        # Correction method selection
        ttk.Label(fitting_frame, text="Correction:").grid(row=1, column=0, sticky='w', pady=5)
        self.correction_var = tk.StringVar(value="look_locker")
        correction_combo = ttk.Combobox(fitting_frame, textvariable=self.correction_var, 
                               values=["none", 
                                      "look_locker", 
                                      "look_locker_t2star", 
                                      "molli_standard", 
                                      "look_locker_inversion", 
                                      "standard_gre", 
                                      "full",
                                      "custom"], width=15)
        correction_combo.grid(row=1, column=1, padx=5, pady=5)
        correction_combo.bind("<<ComboboxSelected>>", self.on_correction_method_changed)

        # Custom correction frame - ONLY VISIBLE WHEN CUSTOM IS SELECTED
        self.custom_frame = ttk.LabelFrame(fitting_frame, text="Custom Corrections", padding=5)
        # Don't pack it initially
        
        # Checkboxes for each correction type
        self.apply_ll_var = tk.BooleanVar(value=True)
        self.apply_t2star_var = tk.BooleanVar(value=False)
        self.apply_sat_var = tk.BooleanVar(value=False)
        self.apply_inv_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(self.custom_frame, text="Look-Locker", variable=self.apply_ll_var, 
               command=self.on_custom_correction_changed).grid(row=0, column=0, sticky='w', padx=5)
        ttk.Checkbutton(self.custom_frame, text="T2* Bias", variable=self.apply_t2star_var,
                       command=self.on_custom_correction_changed).grid(row=0, column=1, sticky='w', padx=5)
        ttk.Checkbutton(self.custom_frame, text="Saturation", variable=self.apply_sat_var,
                       command=self.on_custom_correction_changed).grid(row=1, column=0, sticky='w', padx=5)
        ttk.Checkbutton(self.custom_frame, text="Inversion Eff.", variable=self.apply_inv_var,
                       command=self.on_custom_correction_changed).grid(row=1, column=1, sticky='w', padx=5)
        
        # Additional correction options - ALWAYS VISIBLE
        additional_frame = ttk.LabelFrame(fitting_frame, text="Additional Corrections", padding=5)
        additional_frame.grid(row=3, column=0, columnspan=2, sticky='ew', pady=5)
        
        # T2* value input field
        ttk.Label(additional_frame, text="T2* (ms):").grid(row=0, column=0, sticky='w', pady=2)
        self.t2star_var = tk.StringVar(value="40")  # Default for myocardium
        t2star_entry = ttk.Entry(additional_frame, textvariable=self.t2star_var, width=8)
        t2star_entry.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        t2star_entry.bind("<KeyRelease>", self.auto_recalculate_t1)
        
        # MT correction checkbox
        self.apply_mt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(additional_frame, text="Apply MT Correction (1.5T: ~3-5%)", 
                       variable=self.apply_mt_var,
                       command=self.auto_recalculate_t1).grid(row=1, column=0, columnspan=2, sticky='w', pady=2)
        
        # Cardiac correction checkbox
        self.apply_cardiac_var = tk.BooleanVar(value=False)
        cardiac_check = ttk.Checkbutton(additional_frame, 
                                       text="Cardiac Empirical (+12%)", 
                                       variable=self.apply_cardiac_var,
                                       command=self.auto_recalculate_t1)
        cardiac_check.grid(row=2, column=0, columnspan=2, sticky='w', pady=2)
        
        # Add warning label for cardiac correction
        self.cardiac_warning = ttk.Label(additional_frame, 
                                        text="⚠ Not scientifically justified", 
                                        foreground="orange",
                                        font=("Arial", 8))
        # Initially hidden
        
        # Bind to show/hide warning
        self.apply_cardiac_var.trace('w', self.toggle_cardiac_warning)
        
        # MT analysis buttons
        mt_buttons_frame = ttk.Frame(fitting_frame)
        mt_buttons_frame.grid(row=4, column=0, columnspan=2, pady=5)
        
        self.analyze_mt_button = ttk.Button(mt_buttons_frame, text="Analyze MT Effects", 
                                           command=self.analyze_mt_effects,
                                           state=tk.DISABLED)
        self.analyze_mt_button.pack(side=tk.LEFT, padx=2)
        
        self.enhanced_mt_button = ttk.Button(mt_buttons_frame, text="Enhanced MT Correction", 
                                            command=self.apply_enhanced_mt_correction,
                                            state=tk.DISABLED)
        self.enhanced_mt_button.pack(side=tk.LEFT, padx=2)
        
        # T1 mapping section
        t1_frame = ttk.LabelFrame(self.left_panel, text="T1 Mapping", padding=10)
        t1_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.calc_t1_button = ttk.Button(t1_frame, text="Select ROI & Calculate T1", 
                                   command=self.calculate_t1,
                                   state=tk.DISABLED)
        self.calc_t1_button.pack(fill=tk.X, pady=5)
        
        self.t1_result_var = tk.StringVar(value="No T1 value calculated")
        ttk.Label(t1_frame, textvariable=self.t1_result_var, 
                 wraplength=230).pack(fill=tk.X)

        # Add B1 map integration UI
        self.create_b1_integration_panel()
        
        # Everything else will now be moved to the right panel

    def create_right_panel_content(self):
        # Visualization options
        viz_frame = ttk.LabelFrame(self.right_panel, text="Visualization Options", padding=10)
        viz_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Colormap selection
        ttk.Label(viz_frame, text="Colormap:").grid(row=0, column=0, sticky='w', pady=5)
        self.colormap_var = tk.StringVar(value='turbo')
        self.colormap_combo = ttk.Combobox(viz_frame, textvariable=self.colormap_var, 
                                       values=get_colormap_list(), width=15)
        self.colormap_combo.grid(row=0, column=1, padx=5, pady=5)
        self.colormap_combo.bind("<<ComboboxSelected>>", self.update_visualization)
        
        # Overlay toggle
        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Show Overlay", variable=self.overlay_var,
                       command=self.update_visualization).grid(row=1, column=0, 
                                                          columnspan=2, sticky='w', pady=5)
        
        # Transparency slider
        ttk.Label(viz_frame, text="Transparency:").grid(row=2, column=0, sticky='w', pady=5)
        self.alpha_var = tk.DoubleVar(value=0.7)
        alpha_slider = ttk.Scale(viz_frame, from_=0.1, to=1.0, variable=self.alpha_var,
                              orient=tk.HORIZONTAL, length=120)
        alpha_slider.grid(row=2, column=1, padx=5, pady=5)
        alpha_slider.bind("<ButtonRelease-1>", self.update_visualization)
        
        # T1 range
        ttk.Label(viz_frame, text="T1 Range (ms):").grid(row=3, column=0, sticky='w', pady=5)
        range_frame = ttk.Frame(viz_frame)
        range_frame.grid(row=3, column=1, padx=5, pady=5)
        
        self.t1_min_var = tk.StringVar()
        self.t1_max_var = tk.StringVar()
        t1_min_entry = ttk.Entry(range_frame, textvariable=self.t1_min_var, width=5)
        t1_min_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(range_frame, text="-").pack(side=tk.LEFT, padx=2)
        t1_max_entry = ttk.Entry(range_frame, textvariable=self.t1_max_var, width=5)
        t1_max_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(viz_frame, text="Apply Range", 
                 command=self.update_t1_range).grid(row=4, column=0, columnspan=2, pady=5)
            
        # Add the corrected fit checkbox
        self.show_corrected_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(viz_frame, text="Show Corrected Fit Curve", 
                       variable=self.show_corrected_var,
                       command=self.toggle_corrected_fit).grid(row=5, column=0, 
                                                           columnspan=2, sticky='w', pady=5)
        
        self.show_mt_button = ttk.Button(viz_frame, text="Show MT Analysis", 
                                        command=self.show_mt_analysis,
                                        state=tk.DISABLED)
        self.show_mt_button.grid(row=6, column=0, columnspan=2, pady=5)
        
        # Add quality assessment UI
        self.create_quality_assessment_ui_right()
        
        # Export section
        export_frame = ttk.LabelFrame(self.right_panel, text="Export Results", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.save_image_button = ttk.Button(export_frame, text="Save Current View", 
                                         command=self.save_current_view,
                                         state=tk.DISABLED)
        self.save_image_button.pack(fill=tk.X, pady=5)
        
        self.save_data_button = ttk.Button(export_frame, text="Save T1 Data", 
                                        command=self.save_t1_data,
                                        state=tk.DISABLED)
        self.save_data_button.pack(fill=tk.X, pady=5)
        
        # Add DICOM save button
        self.save_dicom_button = ttk.Button(export_frame, text="Save T1 Map as DICOM", 
                                           command=self.save_t1_map_dicom,
                                           state=tk.DISABLED)
        self.save_dicom_button.pack(fill=tk.X, pady=5)

    # Additional helper methods needed for the UI

    def on_correction_method_changed(self, event=None):
        """Handle correction method dropdown change"""
        method = self.correction_var.get()
        
        if method == "custom":
            # Show custom correction frame
            self.custom_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
            # Build custom correction string
            self.apply_custom_correction()
        else:
            # Hide custom correction frame
            self.custom_frame.grid_forget()
            # Auto-recalculate with new method
            self.auto_recalculate_t1()


    def on_custom_correction_changed(self):
        """Called when custom correction checkboxes change"""
        if self.correction_var.get() == "custom":
            self.apply_custom_correction()


    def apply_custom_correction(self):
        """Apply custom correction based on checkbox selections"""
        # Only apply if "custom" is selected in dropdown
        if self.correction_var.get() != "custom":
            return
            
        # Build custom correction string
        correction_str = "custom:"
        corrections_applied = []
        
        if self.apply_ll_var.get():
            correction_str += "L"
            corrections_applied.append("Look-Locker")
        if self.apply_t2star_var.get():
            correction_str += "T"
            corrections_applied.append("T2* Bias")
        if self.apply_sat_var.get():
            correction_str += "S"
            corrections_applied.append("Saturation")
        if self.apply_inv_var.get():
            correction_str += "B"
            corrections_applied.append("Inversion")
        
        # Store the custom string internally
        self._custom_correction_string = correction_str
        
        # Update status
        if corrections_applied:
            correction_desc = f"Custom: {', '.join(corrections_applied)}"
        else:
            correction_desc = "Custom: No corrections selected"
        
        self.update_status(f"Using {correction_desc}")
        
        # Automatically recalculate T1 if we have ROI data
        self.auto_recalculate_t1()


    def toggle_cardiac_warning(self, *args):
        """Show/hide warning for cardiac correction"""
        if self.apply_cardiac_var.get():
            self.cardiac_warning.grid(row=2, column=2, sticky='w', padx=5)
        else:
            self.cardiac_warning.grid_forget()


    def on_correction_changed(self, *args):
        """Wrapper for compatibility with existing code"""
        self.auto_recalculate_t1()


    # Modified auto_recalculate_t1 to use proper correction method
    def get_active_correction_method(self):
        """Get the currently active correction method string"""
        method = self.correction_var.get()
        
        if method == "custom":
            # Return the custom correction string
            return getattr(self, '_custom_correction_string', 'custom:L')  # Default to LL only
        else:
            # Return the standard method
            return method

    def create_b1_integration_panel(self):
        """Add B1 map integration panel to the UI"""
        # Create the B1 integration frame
        b1_frame = ttk.LabelFrame(self.left_panel, text="B1 Map Integration", padding=10)
        b1_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create notebook for different B1 map sources
        b1_notebook = ttk.Notebook(b1_frame)
        b1_notebook.pack(fill=tk.X, pady=5)
        
        # Tab 1: Load existing B1 map
        load_tab = ttk.Frame(b1_notebook)
        b1_notebook.add(load_tab, text="Load B1 Map")
        
        # B1 map loading button
        self.load_b1_button = ttk.Button(load_tab, text="Load B1 Map File", 
                                        command=self.load_b1_map)
        self.load_b1_button.pack(fill=tk.X, pady=5)
        
        # Tab 2: Create from dual flip angle
        create_tab = ttk.Frame(b1_notebook)
        b1_notebook.add(create_tab, text="Dual FA")
        
        # Create B1 map button
        self.create_b1_button = ttk.Button(create_tab, text="Create B1 Map from Dual FA", 
                                          command=self.create_b1_map_from_fa_series)
        self.create_b1_button.pack(fill=tk.X, pady=5)
        
        # Info label for dual FA method
        info_text = "Uses two image series with\ndifferent flip angles (e.g., 30° and 60°)\nto calculate B1 inhomogeneity"
        ttk.Label(create_tab, text=info_text, font=("Arial", 9), 
                 foreground="gray").pack(pady=5)
        
        # Tab 3: Create from inversion flip angle
        inversion_tab = ttk.Frame(b1_notebook)
        b1_notebook.add(inversion_tab, text="Inversion FA")
        
        # Create B1 map from inversion flip angles button
        self.inversion_b1_button = ttk.Button(inversion_tab, text="Create from Inversion FA", 
                                             command=self.create_b1_map_from_inversion_angles)
        self.inversion_b1_button.pack(fill=tk.X, pady=5)
        
        # Info label for inversion method
        info_text = "Uses two inversion recovery series with different\ninversion pulse flip angles (e.g., 30° and 60°)\nto calculate B1 inhomogeneity"
        ttk.Label(inversion_tab, text=info_text, font=("Arial", 9), 
                 foreground="gray").pack(pady=5)
        
        # B1 map info text (shared)
        self.b1_info_var = tk.StringVar(value="No B1 map loaded")
        ttk.Label(b1_frame, textvariable=self.b1_info_var, 
                 wraplength=230).pack(fill=tk.X, pady=5)
        
        # Add coregistration button
        self.coregister_b1_button = ttk.Button(b1_frame, text="Coregister B1 Map", 
                                             command=self.coregister_b1_map,
                                             state=tk.DISABLED)
        self.coregister_b1_button.pack(fill=tk.X, pady=5)
        
        # Add registration method selection
        ttk.Label(b1_frame, text="Registration Method:").pack(anchor=tk.W)
        self.reg_method_var = tk.StringVar(value="affine")
        reg_method_combo = ttk.Combobox(b1_frame, textvariable=self.reg_method_var, 
                                      values=["rigid", "affine", "bspline", "demons"], width=15)
        reg_method_combo.pack(fill=tk.X, pady=5)
        
        # B1 correction button
        self.apply_b1_button = ttk.Button(b1_frame, text="Apply B1 Correction", 
                                         command=self.apply_b1_correction,
                                         state=tk.DISABLED)
        self.apply_b1_button.pack(fill=tk.X, pady=5)
        
        # Toggle B1 map display
        self.show_b1_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(b1_frame, text="Show B1 Map", variable=self.show_b1_var,
                       command=self.toggle_b1_display).pack(fill=tk.X, pady=5)
        
        # Save B1 map button
        self.save_b1_button = ttk.Button(b1_frame, text="Save B1 Map", 
                                        command=self.save_b1_map,
                                        state=tk.DISABLED)
        self.save_b1_button.pack(fill=tk.X, pady=5)

    def load_dual_inversion_angle_images(fa1_paths, fa2_paths):
        """
        Load two sets of DICOM images with different inversion flip angles
        
        Parameters:
        -----------
        fa1_paths : list
            Paths to DICOM files for first inversion flip angle
        fa2_paths : list
            Paths to DICOM files for second inversion flip angle
                
        Returns:
        --------
        image_fa1 : ndarray
            Average image from first inversion flip angle series
        image_fa2 : ndarray
            Average image from second inversion flip angle series
        fa1 : float
            First inversion flip angle in degrees
        fa2 : float
            Second inversion flip angle in degrees
        inversion_times : list
            List of inversion times for both series
        """
        import numpy as np
        import pydicom
        
        # Load first flip angle series
        images_fa1 = []
        inversion_angles_1 = []
        inversion_times_1 = []
        
        for path in fa1_paths:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            
            # Apply rescale if available
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            img = img * slope + intercept
            
            images_fa1.append(img)
            
            # Try to get inversion angles - might be in private tags or elsewhere
            # For now, assume we know it's 30 degrees
            inversion_angles_1.append(30.0)
            
            # Get inversion time
            if hasattr(ds, 'InversionTime'):
                inversion_times_1.append(float(ds.InversionTime))
        
        # Load second flip angle series
        images_fa2 = []
        inversion_angles_2 = []
        inversion_times_2 = []
        
        for path in fa2_paths:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            
            # Apply rescale if available
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            img = img * slope + intercept
            
            images_fa2.append(img)
            
            # Try to get inversion angles - might be in private tags or elsewhere
            # For now, assume we know it's 60 degrees
            inversion_angles_2.append(60.0)
            
            # Get inversion time
            if hasattr(ds, 'InversionTime'):
                inversion_times_2.append(float(ds.InversionTime))
        
        # Check dimensions match
        shapes_1 = [img.shape for img in images_fa1]
        shapes_2 = [img.shape for img in images_fa2]
        
        if len(set(shapes_1)) > 1 or len(set(shapes_2)) > 1:
            raise ValueError("All images in each series must have the same dimensions")
        
        if shapes_1[0] != shapes_2[0]:
            raise ValueError("Images from both flip angle series must have the same dimensions")
        
        # Average images if multiple slices
        if len(images_fa1) > 1:
            image_fa1 = np.mean(np.stack(images_fa1, axis=0), axis=0)
        else:
            image_fa1 = images_fa1[0]
        
        if len(images_fa2) > 1:
            image_fa2 = np.mean(np.stack(images_fa2, axis=0), axis=0)
        else:
            image_fa2 = images_fa2[0]
        
        # Get inversion flip angles
        if inversion_angles_1:
            fa1 = np.mean(inversion_angles_1)
        else:
            # Default to 30 degrees if not found
            fa1 = 30.0
        
        if inversion_angles_2:
            fa2 = np.mean(inversion_angles_2)
        else:
            # Default to 60 degrees if not found
            fa2 = 60.0
        
        # Ensure flip angles are different
        if abs(fa1 - fa2) < 5:
            raise ValueError(f"Inversion flip angles too similar: {fa1}° and {fa2}°. Need at least 5° difference.")
        
        # Combine inversion times if available
        inversion_times = []
        if inversion_times_1:
            inversion_times.extend(inversion_times_1)
        if inversion_times_2:
            inversion_times.extend(inversion_times_2)
        
        return image_fa1, image_fa2, fa1, fa2, inversion_times

    def calculate_b1_map_from_inversion_angles(image_fa1, image_fa2, fa1_deg, fa2_deg, inversion_times=None, mask_threshold=0.1):
        """
        Calculate B1 map from two images with different inversion pulse flip angles
        
        Parameters:
        -----------
        image_fa1 : ndarray
            Image acquired with first inversion flip angle (typically 30°)
        image_fa2 : ndarray
            Image acquired with second inversion flip angle (typically 60°)
        fa1_deg : float
            First inversion flip angle in degrees
        fa2_deg : float
            Second inversion flip angle in degrees
        inversion_times : list, optional
            Inversion times used (for validation)
        mask_threshold : float
            Threshold for creating signal mask (fraction of max signal)
                
        Returns:
        --------
        b1_map : ndarray
            B1 field map (1.0 = nominal field strength)
        confidence_map : ndarray
            Confidence map based on signal strength
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter
        
        # Ensure images have the same shape
        if image_fa1.shape != image_fa2.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Create mask based on signal intensity
        max_signal = max(np.max(image_fa1), np.max(image_fa2))
        signal_mask = ((image_fa1 > mask_threshold * max_signal) & 
                       (image_fa2 > mask_threshold * max_signal))
        
        # Initialize B1 map and confidence map
        b1_map = np.ones_like(image_fa1, dtype=np.float32)
        confidence_map = np.zeros_like(image_fa1, dtype=np.float32)
        
        # Calculate B1 from inversion recovery signal ratio
        # For inversion recovery with different inversion flip angles:
        # The relationship is: S1/S2 ≈ sin(B1*FA1*π/180) / sin(B1*FA2*π/180)
        
        # Convert flip angles to radians
        fa1_rad = np.radians(fa1_deg)
        fa2_rad = np.radians(fa2_deg)
        
        # Calculate ratio of signals within the mask
        ratio = np.zeros_like(image_fa1, dtype=np.float32)
        ratio[signal_mask] = image_fa1[signal_mask] / image_fa2[signal_mask]
        
        # Apply ratio threshold to avoid division by zero issues
        valid_ratio = (ratio > 0.1) & (ratio < 10) & signal_mask
        
        # For angles 30° and 60°, we can use an approximate analytical solution
        # If S1/S2 = sin(B1*30°)/sin(B1*60°), then:
        # B1 ≈ (2/π) * asin(ratio * sin(60°) / sin(30°))
        
        # Calculate sin values for reference angles
        sin_fa1 = np.sin(fa1_rad)
        sin_fa2 = np.sin(fa2_rad)
        
        # Calculate B1 factor for each valid pixel
        b1_map[valid_ratio] = (2.0 / np.pi) * np.arcsin(
            np.clip(ratio[valid_ratio] * sin_fa2 / sin_fa1, -1.0, 1.0)
        ) * (np.pi / 2.0)  # Normalize to make nominal B1 = 1.0
        
        # Apply smoothing to reduce noise
        b1_smoothed = gaussian_filter(b1_map, sigma=1.0)
        
        # Ensure B1 values are in reasonable range
        b1_smoothed = np.clip(b1_smoothed, 0.5, 1.5)
        
        # Create confidence map based on signal strength and ratio validity
        confidence_map[valid_ratio] = 1.0 - np.abs((ratio[valid_ratio] * sin_fa2 / sin_fa1) - 0.5) / 0.5
        confidence_map = np.clip(confidence_map, 0.0, 1.0)
        
        # Adjust confidence based on signal strength
        avg_signal = (image_fa1 + image_fa2) / 2
        signal_confidence = np.clip(avg_signal / (0.7 * max_signal), 0, 1)
        confidence_map = confidence_map * signal_confidence
        
        return b1_smoothed, confidence_map

    def show_b1_calculation_results(self, image_fa1, image_fa2, b1_map, confidence_map, fa1, fa2):
        """
        Show the results of B1 map calculation with robust error handling
        """
        # Create a new window for results
        result_window = tk.Toplevel(self.root)
        result_window.title("B1 Map Calculation Results")
        result_window.geometry("1200x800")
        
        # Create figure
        fig = plt.Figure(figsize=(12, 8))
        
        # First flip angle image
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(image_fa1, cmap='gray')
        ax1.set_title(f"Inversion FA {fa1:.1f}°")
        ax1.axis('off')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Second flip angle image
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(image_fa2, cmap='gray')
        ax2.set_title(f"Inversion FA {fa2:.1f}°")
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Signal ratio
        ax3 = fig.add_subplot(2, 3, 3)
        ratio_map = np.zeros_like(image_fa1)
        mask = (image_fa2 > 0.1 * np.max(image_fa2))
        # Avoid division by zero
        safe_mask = mask & (image_fa2 > 1e-6)
        if np.any(safe_mask):
            ratio_map[safe_mask] = image_fa1[safe_mask] / image_fa2[safe_mask]
        
        im3 = ax3.imshow(ratio_map, cmap='coolwarm', vmin=0.5, vmax=1.5)
        ax3.set_title("Signal Ratio (FA1/FA2)")
        ax3.axis('off')
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # B1 map
        ax4 = fig.add_subplot(2, 3, 4)
        im4 = ax4.imshow(b1_map, cmap='viridis', vmin=0.7, vmax=1.3)
        ax4.set_title("B1 Map")
        ax4.axis('off')
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('B1 Factor')
        
        # Confidence map
        ax5 = fig.add_subplot(2, 3, 5)
        im5 = ax5.imshow(confidence_map, cmap='plasma', vmin=0, vmax=1)
        ax5.set_title("Confidence Map")
        ax5.axis('off')
        cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.set_label('Confidence')
        
        # Histogram of B1 values - WITH ERROR HANDLING
        ax6 = fig.add_subplot(2, 3, 6)
        valid_b1 = b1_map[confidence_map > 0.5]
        
        # Check if we have any valid B1 values before making histogram
        if len(valid_b1) > 0:
            ax6.hist(valid_b1.flatten(), bins=50, density=True, alpha=0.7, color='blue')
            ax6.axvline(1.0, color='red', linestyle='--', label='Nominal B1')
        else:
            ax6.text(0.5, 0.5, "No high-confidence B1 values", 
                    ha='center', va='center', transform=ax6.transAxes,
                    color='red', fontsize=12)
        
        ax6.set_xlabel('B1 Factor')
        ax6.set_ylabel('Frequency')
        ax6.set_title('B1 Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add statistics text - WITH ERROR HANDLING
        stats_text = f"B1 Statistics:\n"
        
        if len(valid_b1) > 0:
            # Normal case - we have valid data
            stats_text += f"Mean: {np.mean(valid_b1):.3f}\n"
            stats_text += f"Median: {np.median(valid_b1):.3f}\n"
            stats_text += f"Std Dev: {np.std(valid_b1):.3f}\n"
            stats_text += f"Range: [{np.min(valid_b1):.3f}, {np.max(valid_b1):.3f}]"
        else:
            # No valid data case
            stats_text += "No high-confidence B1 values found\n"
            stats_text += "Try lowering the confidence threshold\n"
            stats_text += "or check input images"
        
        ax6.text(0.98, 0.98, stats_text, transform=ax6.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        
        # Add canvas to window
        canvas = FigureCanvasTkAgg(fig, result_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add buttons
        button_frame = ttk.Frame(result_window)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_b1_results():
            file_path = filedialog.asksaveasfilename(
                title="Save B1 Map Results",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"B1 map results saved to {file_path}")
        
        ttk.Button(button_frame, text="Save Figure", 
                  command=save_b1_results).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Close", 
                  command=result_window.destroy).pack(side=tk.RIGHT, padx=5)

    def create_b1_map_from_inversion_angles(self):
        """
        GUI method to create B1 map from two inversion flip angle series
        """
        # Create dialog for selecting two series
        dialog = tk.Toplevel(self.root)
        dialog.title("Create B1 Map from Inversion Flip Angles")
        dialog.geometry("500x450")
        
        # Instructions
        ttk.Label(dialog, text="Select DICOM files for two different inversion flip angles (30° and 60°)",
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Add information label
        info_text = ("This method uses two inversion recovery series with different\n"
                    "inversion pulse flip angles (typically 30° and 60°) to calculate\n"
                    "B1 inhomogeneity. The imaging flip angle is typically 90°.")
        ttk.Label(dialog, text=info_text, font=("Arial", 10), foreground="blue").pack(pady=5)
        
        # First flip angle series
        fa1_frame = ttk.LabelFrame(dialog, text="Inversion Flip Angle 1 (30°)", padding=10)
        fa1_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.fa1_files = []
        self.fa1_info_var = tk.StringVar(value="No files selected")
        ttk.Label(fa1_frame, textvariable=self.fa1_info_var).pack()
        
        def select_fa1_files():
            files = list(filedialog.askopenfilenames(
                title="Select 30° Inversion Flip Angle DICOM Files",
                filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
            ))
            if files:
                self.fa1_files = files
                self.fa1_info_var.set(f"{len(files)} files selected (Inversion FA: 30°)")
        
        ttk.Button(fa1_frame, text="Select Files", command=select_fa1_files).pack(pady=5)
        
        # Second flip angle series
        fa2_frame = ttk.LabelFrame(dialog, text="Inversion Flip Angle 2 (60°)", padding=10)
        fa2_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.fa2_files = []
        self.fa2_info_var = tk.StringVar(value="No files selected")
        ttk.Label(fa2_frame, textvariable=self.fa2_info_var).pack()
        
        def select_fa2_files():
            files = list(filedialog.askopenfilenames(
                title="Select 60° Inversion Flip Angle DICOM Files",
                filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
            ))
            if files:
                self.fa2_files = files
                self.fa2_info_var.set(f"{len(files)} files selected (Inversion FA: 60°)")
        
        ttk.Button(fa2_frame, text="Select Files", command=select_fa2_files).pack(pady=5)
        
        # Angle input fields
        angle_frame = ttk.Frame(dialog)
        angle_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(angle_frame, text="Actual Inversion Flip Angles:").grid(row=0, column=0, pady=5)
        
        fa1_var = tk.StringVar(value="30")
        fa2_var = tk.StringVar(value="60")
        
        ttk.Label(angle_frame, text="Angle 1:").grid(row=1, column=0)
        ttk.Entry(angle_frame, textvariable=fa1_var, width=5).grid(row=1, column=1)
        ttk.Label(angle_frame, text="°").grid(row=1, column=2)
        
        ttk.Label(angle_frame, text="Angle 2:").grid(row=2, column=0)
        ttk.Entry(angle_frame, textvariable=fa2_var, width=5).grid(row=2, column=1)
        ttk.Label(angle_frame, text="°").grid(row=2, column=2)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def process_b1_map():
            if not self.fa1_files or not self.fa2_files:
                messagebox.showwarning("Missing Files", 
                                     "Please select files for both inversion flip angle series")
                return
            
            try:
                self.update_status("Loading inversion flip angle images...")
                
                # Get actual angles from input fields
                try:
                    angle1 = float(fa1_var.get())
                    angle2 = float(fa2_var.get())
                    
                    if angle1 <= 0 or angle2 <= 0 or angle1 >= 180 or angle2 >= 180:
                        raise ValueError("Invalid flip angles")
                except:
                    messagebox.showwarning("Invalid Angles", 
                                         "Please enter valid flip angles (0-180°)")
                    return
                
                # Load the images with actual angles
                image_fa1, image_fa2, _, _, inversion_times = load_dual_inversion_angle_images(
                    self.fa1_files, self.fa2_files
                )
                
                self.update_status(f"Calculating B1 map from inversion flip angles {angle1:.1f}° and {angle2:.1f}°...")
                
                # Calculate B1 map
                self.b1_map, confidence_map = calculate_b1_map_from_inversion_angles(
                    image_fa1, image_fa2, angle1, angle2, inversion_times
                )
                
                # Update UI
                self.b1_info_var.set(f"B1 map calculated from inversion flip angles\n"
                                   f"FA1: {angle1:.1f}°, FA2: {angle2:.1f}°\n"
                                   f"Size: {self.b1_map.shape[0]}×{self.b1_map.shape[1]} px\n"
                                   f"Range: {np.min(self.b1_map):.2f}-{np.max(self.b1_map):.2f}")
                
                # Enable B1 correction button if T1 map exists
                if hasattr(self, 't1_map') and self.t1_map is not None:
                    self.apply_b1_button.configure(state=tk.NORMAL)
                
                # Enable coregistration button
                if hasattr(self, 'coregister_b1_button'):
                    self.coregister_b1_button.configure(state=tk.NORMAL)
                
                # Enable save B1 map button
                if hasattr(self, 'save_b1_button'):
                    self.save_b1_button.configure(state=tk.NORMAL)
                
                # Show B1 map visualization
                self.show_b1_calculation_results(image_fa1, image_fa2, 
                                               self.b1_map, confidence_map,
                                               angle1, angle2)
                
                self.update_status("B1 map calculated successfully")
                dialog.destroy()
                
            except Exception as e:
                self.update_status("Error calculating B1 map")
                show_error_dialog("B1 Calculation Error", 
                                "Error calculating B1 map from inversion flip angles", e)
        
        ttk.Button(button_frame, text="Calculate B1 Map", 
                  command=process_b1_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def create_b1_map_from_inversion_angles(self):
        """
        GUI method to create B1 map from two inversion flip angle series
        """
        # Create dialog for selecting two series
        dialog = tk.Toplevel(self.root)
        dialog.title("Create B1 Map from Inversion Flip Angles")
        dialog.geometry("500x450")
        
        # Instructions
        ttk.Label(dialog, text="Select DICOM files for two different inversion flip angles (30° and 60°)",
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Add information label
        info_text = ("This method uses two inversion recovery series with different\n"
                    "inversion pulse flip angles (typically 30° and 60°) to calculate\n"
                    "B1 inhomogeneity. The imaging flip angle is typically 90°.")
        ttk.Label(dialog, text=info_text, font=("Arial", 10), foreground="blue").pack(pady=5)
        
        # First flip angle series
        fa1_frame = ttk.LabelFrame(dialog, text="Inversion Flip Angle 1 (30°)", padding=10)
        fa1_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.fa1_files = []
        self.fa1_info_var = tk.StringVar(value="No files selected")
        ttk.Label(fa1_frame, textvariable=self.fa1_info_var).pack()
        
        def select_fa1_files():
            files = list(filedialog.askopenfilenames(
                title="Select 30° Inversion Flip Angle DICOM Files",
                filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
            ))
            if files:
                self.fa1_files = files
                self.fa1_info_var.set(f"{len(files)} files selected (Inversion FA: 30°)")
        
        ttk.Button(fa1_frame, text="Select Files", command=select_fa1_files).pack(pady=5)
        
        # Second flip angle series
        fa2_frame = ttk.LabelFrame(dialog, text="Inversion Flip Angle 2 (60°)", padding=10)
        fa2_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.fa2_files = []
        self.fa2_info_var = tk.StringVar(value="No files selected")
        ttk.Label(fa2_frame, textvariable=self.fa2_info_var).pack()
        
        def select_fa2_files():
            files = list(filedialog.askopenfilenames(
                title="Select 60° Inversion Flip Angle DICOM Files",
                filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
            ))
            if files:
                self.fa2_files = files
                self.fa2_info_var.set(f"{len(files)} files selected (Inversion FA: 60°)")
        
        ttk.Button(fa2_frame, text="Select Files", command=select_fa2_files).pack(pady=5)
        
        # Angle input fields
        angle_frame = ttk.Frame(dialog)
        angle_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(angle_frame, text="Actual Inversion Flip Angles:").grid(row=0, column=0, pady=5)
        
        fa1_var = tk.StringVar(value="30")
        fa2_var = tk.StringVar(value="60")
        
        ttk.Label(angle_frame, text="Angle 1:").grid(row=1, column=0)
        ttk.Entry(angle_frame, textvariable=fa1_var, width=5).grid(row=1, column=1)
        ttk.Label(angle_frame, text="°").grid(row=1, column=2)
        
        ttk.Label(angle_frame, text="Angle 2:").grid(row=2, column=0)
        ttk.Entry(angle_frame, textvariable=fa2_var, width=5).grid(row=2, column=1)
        ttk.Label(angle_frame, text="°").grid(row=2, column=2)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def process_b1_map():
            if not self.fa1_files or not self.fa2_files:
                messagebox.showwarning("Missing Files", 
                                     "Please select files for both inversion flip angle series")
                return
            
            try:
                self.update_status("Loading inversion flip angle images...")
                
                # Get actual angles from input fields
                try:
                    angle1 = float(fa1_var.get())
                    angle2 = float(fa2_var.get())
                    
                    if angle1 <= 0 or angle2 <= 0 or angle1 >= 180 or angle2 >= 180:
                        raise ValueError("Invalid flip angles")
                except:
                    messagebox.showwarning("Invalid Angles", 
                                         "Please enter valid flip angles (0-180°)")
                    return
                
                # Load the images with actual angles
                image_fa1, image_fa2, _, _, inversion_times = load_dual_inversion_angle_images(
                    self.fa1_files, self.fa2_files
                )
                
                self.update_status(f"Calculating B1 map from inversion flip angles {angle1:.1f}° and {angle2:.1f}°...")
                
                # Calculate B1 map
                self.b1_map, confidence_map = calculate_b1_map_from_inversion_angles(
                    image_fa1, image_fa2, angle1, angle2, inversion_times
                )
                
                # Update UI
                self.b1_info_var.set(f"B1 map calculated from inversion flip angles\n"
                                   f"FA1: {angle1:.1f}°, FA2: {angle2:.1f}°\n"
                                   f"Size: {self.b1_map.shape[0]}×{self.b1_map.shape[1]} px\n"
                                   f"Range: {np.min(self.b1_map):.2f}-{np.max(self.b1_map):.2f}")
                
                # Enable B1 correction button if T1 map exists
                if hasattr(self, 't1_map') and self.t1_map is not None:
                    self.apply_b1_button.configure(state=tk.NORMAL)
                
                # Enable coregistration button
                if hasattr(self, 'coregister_b1_button'):
                    self.coregister_b1_button.configure(state=tk.NORMAL)
                
                # Enable save B1 map button
                if hasattr(self, 'save_b1_button'):
                    self.save_b1_button.configure(state=tk.NORMAL)
                
                # Show B1 map visualization
                self.show_b1_calculation_results(image_fa1, image_fa2, 
                                               self.b1_map, confidence_map,
                                               angle1, angle2)
                
                self.update_status("B1 map calculated successfully")
                dialog.destroy()
                
            except Exception as e:
                self.update_status("Error calculating B1 map")
                show_error_dialog("B1 Calculation Error", 
                                "Error calculating B1 map from inversion flip angles", e)
        
        ttk.Button(button_frame, text="Calculate B1 Map", 
                  command=process_b1_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    
    def create_quality_assessment_ui_right(self):
        """Add quality assessment UI components to the right panel"""
        # Create the quality assessment frame
        quality_frame = ttk.LabelFrame(self.right_panel, text="Quality Assessment", padding=10)
        quality_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Quality map button
        self.quality_maps_button = ttk.Button(quality_frame, text="Show Quality Maps", 
                                            command=self.show_quality_maps,
                                            state=tk.DISABLED)
        self.quality_maps_button.pack(fill=tk.X, pady=5)
        
        # Quality report button
        self.quality_report_button = ttk.Button(quality_frame, text="Generate Quality Report", 
                                              command=self.show_quality_report,
                                              state=tk.DISABLED)
        self.quality_report_button.pack(fill=tk.X, pady=5)
        
        # Display buttons state depends on T1 map existence
        if hasattr(self, 't1_map') and self.t1_map is not None:
            self.quality_maps_button.configure(state=tk.NORMAL)
            self.quality_report_button.configure(state=tk.NORMAL)

    def create_visualization_area(self):
        # Create matplotlib figure for visualization
        self.fig = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.center_panel)  # Changed from self.right_panel
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add initial message
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Load DICOM files to begin", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        self.canvas.draw()
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load DICOM Files", command=self.load_dicom_files)
        file_menu.add_command(label="Load B1 Map", command=self.load_b1_map)
        file_menu.add_separator()
        file_menu.add_command(label="Save Current View", command=self.save_current_view)
        file_menu.add_command(label="Save T1 Data", command=self.save_t1_data)
        file_menu.add_command(label="Save T1 Map as DICOM", command=self.save_t1_map_dicom)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Calculate T1", command=self.calculate_t1)
        tools_menu.add_command(label="Apply B1 Correction", command=self.apply_b1_correction)
        tools_menu.add_separator()
        tools_menu.add_command(label="Show Quality Maps", command=self.show_quality_maps)
        tools_menu.add_command(label="Generate Quality Report", command=self.show_quality_report)
        tools_menu.add_separator()
        tools_menu.add_command(label="Reset View", command=self.reset_visualization)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

# Core Application Methods - Add these to the LeanT1MappingApp class

    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()
        
    def load_dicom_files(self):
        try:
            self.update_status("Selecting DICOM files...")

            # Clear previous data
            self.image_stack = None
            self.corrected_stack = None
            self.tis = None
            self.sequence_params = {}
            self.t1_result = None
            self.roi_coords = None
            self.processed_signal = None
            self.fitted_curve = None
            self.t1_map = None
            self.pixel_cache = {}
            
            # Create file dialog for selecting DICOM files (no filter)
            files = list(filedialog.askopenfilenames(
                title="Select IR DICOM Files"
            ))
            
            if not files:
                self.update_status("No files selected")
                return
            
            self.update_status(f"Loading {len(files)} files...")
            
            # Use the robust loader with sequence parameters
            self.image_stack, self.tis, self.sequence_params = load_ir_dicoms_robust(files)
            self.dicom_files = files
            
            # Show progress dialog for motion correction
            correction_window = tk.Toplevel(self.root)
            correction_window.title("Motion Correction")
            correction_window.geometry("400x150")
            
            ttk.Label(correction_window, text="Applying demons motion correction...", 
                     font=("Arial", 12)).pack(pady=10)
            
            progress = ttk.Progressbar(correction_window, length=300, mode="indeterminate")
            progress.pack(pady=10)
            progress.start()
            
            # Update UI before starting computation
            correction_window.update()
            self.root.update_idletasks()
            
            # Apply motion correction
            try:
                self.corrected_stack = apply_demons_motion_correction(self.image_stack)
                
                # Update UI
                self.file_info_var.set(f"Loaded {len(files)} files\n"
                                     f"TI values: {', '.join([f'{ti:.1f}' for ti in self.tis])} ms\n"
                                     f"Image size: {self.image_stack.shape[1]}×{self.image_stack.shape[2]} px\n"
                                     f"TR: {self.sequence_params.get('TR', 'Unknown')} ms\n"
                                     f"TE: {self.sequence_params.get('TE', 'Unknown')} ms\n" 
                                     f"Flip Angle: {self.sequence_params.get('FlipAngle', 'Unknown')}°")
                
                # Enable T1 button
                self.calc_t1_button.configure(state=tk.NORMAL)
                # After enabling the T1 button (around line 1680)
                self.calc_t1_button.configure(state=tk.NORMAL)

                # Enable MT analysis button
                if hasattr(self, 'analyze_mt_button'):
                    self.analyze_mt_button.configure(state=tk.NORMAL)
                # Show image stack overview
                self.show_stack_overview()
                
                self.update_status("Files loaded successfully with motion correction")
            except Exception as e:
                self.update_status("Error in motion correction")
                show_error_dialog("Motion Correction Error", "Error applying motion correction", e)
                
                # Use original images if motion correction fails
                self.corrected_stack = self.image_stack
                
                # Still enable T1 button
                self.calc_t1_button.configure(state=tk.NORMAL)
            
            # Close progress dialog
            progress.stop()
            correction_window.destroy()
            
        except Exception as e:
            self.update_status("Error loading files")
            show_error_dialog("File Loading Error", "Error loading DICOM files", e)

    def show_stack_overview(self):
        """Show overview of the loaded image stack"""
        if self.corrected_stack is None:
            return
        
        # Clear the figure
        self.fig.clear()
        
        # Create a grid of subplot images
        n_images = min(len(self.tis), 9)  # Show at most 9 images
        rows = int(np.ceil(np.sqrt(n_images)))
        cols = int(np.ceil(n_images / rows))
        
        for i in range(n_images):
            ax = self.fig.add_subplot(rows, cols, i+1)
            ax.imshow(self.corrected_stack[i], cmap='gray')
            ax.set_title(f"TI = {self.tis[i]:.1f} ms")
            ax.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()

    def update_t1_range(self):
        """Update T1 range for colormap"""
        try:
            min_val = float(self.t1_min_var.get())
            max_val = float(self.t1_max_var.get())
            
            if min_val >= max_val:
                messagebox.showwarning("Invalid Range", "Minimum T1 must be less than maximum T1")
                return
            
            self.t1_min = min_val
            self.t1_max = max_val
            
            # Update visualization
            self.update_visualization()
            
        except ValueError:
            messagebox.showwarning("Invalid Value", "Please enter valid numbers for T1 range")

    def show_about(self):
        """Show about dialog with GRE MOLLI specifics"""
        about_window = tk.Toplevel(self.root)
        about_window.title("About GRE MOLLI T1 Mapping Tool")
        about_window.geometry("500x350")
        about_window.transient(self.root)
        about_window.grab_set()
        
        ttk.Label(about_window, text="Enhanced GRE MOLLI T1 Mapping Tool", 
                 font=("Arial", 16, "bold")).pack(pady=10)
        
        ttk.Label(about_window, text="Version 2.0.0").pack()
        
        ttk.Label(about_window, text="A comprehensive tool for T1 mapping from GRE MOLLI sequences with physically constrained fitting and advanced corrections",
                 wraplength=400, justify=tk.CENTER).pack(pady=10)
        
        ttk.Label(about_window, text="Key Features:",
                 font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=20)
        
        features_text = """
        • Physically constrained T1 fitting for GRE MOLLI
        • Comprehensive correction pipeline:
          - Look-Locker correction
          - GRE-specific saturation effects
          - T2* influence correction
          - Inversion efficiency correction
        • B1 map integration for spatial correction
        • Advanced quality assessment
        • Interactive visualization
        """
        
        ttk.Label(about_window, text=features_text,
                 justify=tk.LEFT).pack(anchor=tk.W, padx=40)
        
        import datetime
        ttk.Label(about_window, text=f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d')}",
                 font=("Arial", 8)).pack(side=tk.BOTTOM, pady=10)
        
        ttk.Button(about_window, text="Close", 
                  command=about_window.destroy).pack(side=tk.BOTTOM, pady=10)

    def save_current_view(self):
        """Save the current visualization to an image file"""
        if not hasattr(self, 'fig') or self.fig is None:
            messagebox.showinfo("Info", "No visualization to save")
            return
        
        try:
            # Ask for the file path
            file_path = filedialog.asksaveasfilename(
                title="Save Current View",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPG files", "*.jpg"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                self.update_status("Save cancelled")
                return
            
            # Save the figure
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            self.update_status(f"Image saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Visualization saved to {file_path}")
            
        except Exception as e:
            self.update_status("Error saving image")
            show_error_dialog("Save Error", "Error saving visualization", e)


    def analyze_mt_effects(self):
            """Analyze MT effects in the loaded data"""
            if self.corrected_stack is None or self.tis is None:
                messagebox.showinfo("Info", "Please load DICOM files first")
                return
            
            try:
                self.update_status("Analyzing MT effects...")
                
                # Show ROI selection for MT analysis
                avg_image = np.mean(self.corrected_stack, axis=0)
                roi_coords = enhanced_roi_selection(avg_image, title="Select ROI for MT Analysis")
                
                if not roi_coords:
                    self.update_status("MT analysis cancelled")
                    return
                
                x1, x2, y1, y2 = roi_coords
                roi_mask = np.zeros(avg_image.shape, dtype=bool)
                roi_mask[y1:y2, x1:x2] = True
                
                # FIXED: Call with correct parameters
                mt_map, confidence_map, residual_stats = analyze_ti_signals_for_mt(
                    self.tis, self.corrected_stack, roi_mask, enhanced_detection=False
                )
                
                # Store MT results
                self.mt_map = mt_map
                self.mt_confidence_map = confidence_map
                self.mt_residual_stats = residual_stats
                
                # Show MT analysis results
                messagebox.showinfo("MT Analysis Results",
                                  f"MT detected in {residual_stats['detected_fraction']*100:.1f}% of pixels\n"
                                  f"Mean residual correlation: {residual_stats['mean_correlation']:.3f}\n"
                                  f"Total pixels analyzed: {residual_stats['total_pixels_analyzed']}")
                
                # Enable MT correction in UI
                self.apply_mt_var.set(True)
                
                # Enable show MT button
                if self.show_mt_button:
                    self.show_mt_button.configure(state=tk.NORMAL)
                
                self.update_status("MT analysis complete")
                
            except Exception as e:
                self.update_status("Error in MT analysis")
                show_error_dialog("MT Analysis Error", "Error analyzing MT effects", e)
            
# T1 Calculation and Visualization Methods - Add these to the LeanT1MappingApp class

    def calculate_t1(self):
        """T1 calculation with ROI selection and comprehensive correction for GRE MOLLI"""
        if self.corrected_stack is None:
            messagebox.showinfo("Info", "Please load DICOM files first")
            return
        
        try:
            self.update_status("Selecting ROI for T1 calculation...")
            
            # Show average image for ROI selection
            avg_image = np.mean(self.corrected_stack, axis=0)
            
            # Use enhanced ROI selection
            self.roi_coords = enhanced_roi_selection(
                avg_image,
                title="Select ROI for T1 Calculation"
            )
            
            if not self.roi_coords:
                self.update_status("ROI selection cancelled")
                return
            
            # Extract ROI data
            x1, x2, y1, y2 = self.roi_coords
            roi_data = self.corrected_stack[:, y1:y2, x1:x2]
            
            # Calculate mean signal across ROI
            mean_signal = np.mean(roi_data, axis=(1,2))
            
            # Get parameters needed for corrections
            flip_angle = self.sequence_params.get('FlipAngle')
            tr = self.sequence_params.get('TR')
            te = self.sequence_params.get('TE')
            
            # Try to get T2* from UI
            try:
                t2star = float(self.t2star_var.get()) if hasattr(self, 't2star_var') else 40
            except:
                t2star = 40  # Default for myocardium
            
            # Calculate inversion efficiency
            inversion_efficiency = 1.0  # Default perfect inversion
            if flip_angle is not None:
                inversion_efficiency = calculate_inversion_efficiency(flip_angle)
                if inversion_efficiency < 0.8:
                    messagebox.showinfo("Inversion Efficiency", 
                                      f"Calculated inversion efficiency: {inversion_efficiency:.2f}\n"
                                      f"This correction will be applied to T1 values.")
            
            # Get T1 for the ROI average with physically constrained fit
            try:
                # Use our new physically constrained fit
                popt, self.processed_signal, self.fitted_curve, fit_status = physically_constrained_t1_fit(
                    mean_signal, self.tis, 
                    flip_angle=flip_angle,
                    tr=tr,
                    te=te,
                    t2star=t2star,
                    inversion_efficiency=inversion_efficiency,
                    timeout=1.0,  # Longer timeout for mean ROI
                    model=self.model_var.get(),
                    correction_method=self.correction_var.get(),
                    apply_cardiac_correction=self.apply_cardiac_var.get(),
                    apply_mt_correction=self.apply_mt_var.get()
                )
                
                a, b, self.t1_value = popt
                
            except Exception as e:
                print(f"Error in robust fitting: {e}")
                # Fall back to default values if all fitting fails
                a = np.max(mean_signal)
                b = np.ptp(mean_signal)
                self.t1_value = 1000.0  # Default in ms
                self.processed_signal = mean_signal
                self.fitted_curve = ir_model_simple(self.tis, a, b, self.t1_value)
                fit_status = {'fallback_used': True, 'status': 'error'}
            
            # Calculate average image for visualization
            self.avg_image = np.mean(self.corrected_stack, axis=0)
            
            # Initialize t1_map with the same shape as the ROI
            roi_height, roi_width = roi_data.shape[1:]
            self.t1_map = np.zeros((roi_height, roi_width)) * np.nan
            
            # Calculate T1 for each pixel in the ROI
            self.update_status("Calculating T1 map...")
            
            # Show progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("T1 Calculation")
            progress_window.geometry("400x150")
            
            ttk.Label(progress_window, text="Calculating T1 map...", 
                     font=("Arial", 12)).pack(pady=10)
            
            progress = ttk.Progressbar(progress_window, length=300, mode="determinate")
            progress.pack(pady=10)
            progress["maximum"] = roi_height
            
            self.pixel_cache = {}  # Clear pixel cache
            
            # Calculate T1 for each pixel in the ROI
            for y in range(roi_height):
                for x in range(roi_width):
                    # Extract signal for this pixel
                    pixel_signal = roi_data[:, y, x].copy()
                    
                    # Skip pixels with low signal
                    if np.max(pixel_signal) < 1e-6 or np.allclose(pixel_signal, 0):
                        self.t1_map[y, x] = np.nan
                    else:
                        try:
                            # Try physically constrained fitting with shorter timeout
                            popt, processed_signal, fitted_curve, fit_status = physically_constrained_t1_fit(
                                pixel_signal, self.tis,
                                flip_angle=flip_angle,
                                tr=tr,
                                te=te,
                                t2star=t2star,
                                inversion_efficiency=inversion_efficiency,
                                timeout=0.3,  # Shorter timeout for individual pixels
                                model=self.model_var.get(),
                                correction_method=self.correction_var.get(),
                                apply_cardiac_correction=self.apply_cardiac_var.get(),
                                apply_mt_correction=self.apply_mt_var.get() 
                            )
                            a, b, t1 = popt
                            
                            # Store T1 value (already corrected)
                            self.t1_map[y, x] = t1
                            
                            # Calculate the apparent T1 (uncorrected) from the corrected T1
                            correction_factor = fit_status.get('correction_factors', {}).get('total', 1.0)
                            t1_apparent = t1 / correction_factor  # Calculate apparent T1 from corrected T1
                            
                            # Cache comprehensive pixel information
                            self.pixel_cache[(x, y)] = {
                                'a': a,
                                'b': b,
                                't1': t1,  # corrected T1
                                't1_apparent': t1_apparent,  # uncorrected T1
                                'signal': processed_signal,
                                'fit': fitted_curve,
                                'model': self.model_var.get(),
                                'correction': self.correction_var.get(),
                                'r_squared': fit_status.get('r_squared'),
                                'rmse': fit_status.get('rmse'),
                                'fallback_used': fit_status.get('fallback_used', False),
                                'inversion_efficiency': inversion_efficiency,
                                'correction_factors': fit_status.get('correction_factors', {'total': 1.0}),
                                'mt_detected': fit_status.get('mt_detected', False),  # Add this
                                'mt_fraction': fit_status.get('mt_fraction', 0.0),   # Add this
                                'mt_correlation': fit_status.get('mt_correlation', 0.0)  # Add this
                            }
                            
                        except Exception as e:
                            print(f"Error in pixel fit at ({x}, {y}): {e}")
                            self.t1_map[y, x] = np.nan
                
                # Update progress
                progress["value"] = y + 1
                progress_window.update()
            
            # Close progress window
            progress_window.destroy()
            
            # Calculate statistics
            valid_t1s = self.t1_map[~np.isnan(self.t1_map)]
            if len(valid_t1s) > 0:
                t1_mean = np.mean(valid_t1s)
                t1_median = np.median(valid_t1s)
                t1_std = np.std(valid_t1s)
                success_rate = len(valid_t1s) / (roi_width * roi_height) * 100
            else:
                t1_mean = t1_median = t1_std = np.nan
                success_rate = 0.0
            
            # Update UI with values in ms
            cardiac_note = " (with cardiac correction)" if self.apply_cardiac_var.get() else ""
            self.t1_result_var.set(f"T1 = {self.t1_value:.1f} ms (Mean ROI){cardiac_note}\n"
                                 f"Median: {t1_median:.1f} ms\n"
                                 f"StdDev: {t1_std:.1f} ms\n"
                                 f"Success: {success_rate:.1f}%")
            
            # Set default T1 range based on valid T1s
            if len(valid_t1s) > 0:
                self.t1_min = np.percentile(valid_t1s, 1)
                self.t1_max = np.percentile(valid_t1s, 99)
                self.t1_min_var.set(f"{self.t1_min:.1f}")
                self.t1_max_var.set(f"{self.t1_max:.1f}")
            
            # Show T1 map
            self.show_t1_map_with_overlay()
                     
            # Enable export buttons
            self.save_image_button.configure(state=tk.NORMAL)
            self.save_data_button.configure(state=tk.DISABLED)  # Will enable after adding save_t1_data method
            
            # Enable DICOM save button if it exists
            if hasattr(self, 'save_dicom_button'):
                self.save_dicom_button.configure(state=tk.NORMAL)
            
            # Enable quality assessment buttons
            if hasattr(self, 'quality_maps_button'):
                self.quality_maps_button.configure(state=tk.NORMAL)
            if hasattr(self, 'quality_report_button'):
                self.quality_report_button.configure(state=tk.NORMAL)
            
            # Enable B1 correction if B1 map is available
            if hasattr(self, 'b1_map') and self.b1_map is not None:
                self.apply_b1_button.configure(state=tk.NORMAL)

            # Enable enhanced MT correction button
            if hasattr(self, 'enhanced_mt_button'):
                self.enhanced_mt_button.configure(state=tk.NORMAL)
            
            self.update_status(f"T1 calculated: {self.t1_value:.1f} ms (Mean ROI){cardiac_note}")
            
        except Exception as e:
            self.update_status("Error calculating T1")
            show_error_dialog("T1 Calculation Error", "Error calculating T1", e)
            print(f"Calculation error: {e}")
            traceback.print_exc()

    def show_t1_map_with_overlay(self):
        """Show T1 map with overlay on original image with fit panel side by side"""
        if self.t1_map is None or self.avg_image is None:
            return
        
        # Clear the figure
        self.fig.clear()
        
        # Create subplot for T1 map view (left side)
        self.main_ax = self.fig.add_subplot(121)  # 1 row, 2 columns, 1st position
        
        # Get ROI coordinates
        x1, x2, y1, y2 = self.roi_coords
        
        # Get display mode based on checkbox
        self.colormap_name = self.colormap_var.get()
        self.overlay_alpha = self.alpha_var.get()
        self.show_overlay = self.overlay_var.get()
        
        # Display only the ROI portion of the original average image
        roi_avg_image = self.avg_image[y1:y2, x1:x2]
        self.background_img = self.main_ax.imshow(roi_avg_image, cmap='gray')
        
        # Create and display the colorized T1 map overlay if enabled
        if self.show_overlay:
            # Apply colormap directly to the T1 map
            t1_rgba = apply_colormap(
                self.t1_map, 
                colormap_name=self.colormap_name,
                min_val=self.t1_min,
                max_val=self.t1_max,
                alpha=self.overlay_alpha
            )
            
            # Display overlay
            self.overlay_img = self.main_ax.imshow(t1_rgba)
        
            # Add colorbar
            self.colorbar = self.fig.colorbar(self.overlay_img, ax=self.main_ax)
            self.colorbar.set_label('T1 (ms)')  # Keep units as ms
        
        # Add title with ROI dimensions and T1 value in ms
        roi_width = x2 - x1
        roi_height = y2 - y1
        self.main_ax.set_title(f"T1 Map - ROI: {roi_width}×{roi_height} px\nMean T1: {self.t1_value:.2f} ms")
        
        # Remove axis labels
        self.main_ax.axis('off')
        
        # Add cursor position indicator on image
        self.cursor_point, = self.main_ax.plot([], [], 'o', color='lime', markersize=8)
        
        # Setup the pixel info text box
        self.cursor_info = self.main_ax.text(0.02, 0.98, "", 
                                           transform=self.main_ax.transAxes,
                                           va='top', ha='left',
                                           bbox=dict(facecolor='white', alpha=0.8))
        
        # Create fit curve panel (right side)
        self.curve_ax = self.fig.add_subplot(122)  # 1 row, 2 columns, 2nd position
        self.curve_ax.set_title("Pixel T1 Fit")
        self.curve_ax.set_xlabel("Inversion Time (ms)")
        self.curve_ax.set_ylabel("Signal")
        self.curve_ax.grid(True, alpha=0.3)
        
        # Add "No data" text to center of fit panel initially
        self.no_data_text = self.curve_ax.text(0.5, 0.5, "No data selected",
                                            transform=self.curve_ax.transAxes,
                                            va='center', ha='center',
                                            fontsize=12, color='gray')
        
        # Setup mouse events
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Add instruction text at the bottom
        self.fig.text(0.5, 0.01, 
                     "Hover over T1 map to see pixel values and fit curves",
                     ha='center', fontsize=10,
                     bbox=dict(facecolor='lightgray', alpha=0.5))
        
        self.fig.tight_layout()
        self.canvas.draw()

    def show_mt_analysis(self):
        """Show MT analysis results"""
        if not hasattr(self, 't1_map') or self.t1_map is None:
            messagebox.showinfo("Info", "Calculate T1 first")
            return
        
        try:
            self.update_status("Generating MT analysis visualization...")
            
            # Create MT map from pixel cache
            mt_map = np.zeros_like(self.t1_map)
            confidence_map = np.zeros_like(self.t1_map)
            
            for (x, y), data in self.pixel_cache.items():
                if 0 <= y < mt_map.shape[0] and 0 <= x < mt_map.shape[1]:
                    mt_map[y, x] = data.get('mt_fraction', 0.0)
                    confidence_map[y, x] = abs(data.get('mt_correlation', 0.0))
            
            # Apply MT correction to generate corrected T1 map
            t1_corrected = apply_mt_correction_to_t1map(self.t1_map, mt_map)
            
            # Show comprehensive MT visualization
            plot_mt_effects(
                self.t1_map, 
                t1_corrected, 
                mt_map, 
                confidence_map,
                roi_coords=self.roi_coords,
                save_path=None
            )
            
            self.update_status("MT analysis visualization complete")
            
        except Exception as e:
            self.update_status("Error in MT visualization")
            show_error_dialog("MT Visualization Error", "Error showing MT analysis", e)
    def reset_visualization(self):
        """Reset visualization to default state"""
        if self.t1_map is None:
            messagebox.showinfo("Info", "No T1 map to reset")
            return
            
        try:
            # Reset visualization parameters to defaults
            self.colormap_name = 'turbo'
            self.colormap_var.set(self.colormap_name)
            
            self.overlay_alpha = 0.7
            self.alpha_var.set(self.overlay_alpha)
            
            self.show_overlay = True
            self.overlay_var.set(self.show_overlay)
            
            # Reset T1 range to calculated percentiles
            valid_t1s = self.t1_map[~np.isnan(self.t1_map)]
            if len(valid_t1s) > 0:
                self.t1_min = np.percentile(valid_t1s, 1)
                self.t1_max = np.percentile(valid_t1s, 99)
                self.t1_min_var.set(f"{self.t1_min:.1f}")
                self.t1_max_var.set(f"{self.t1_max:.1f}")
            
            # Update the visualization
            if hasattr(self, 'main_ax'):
                self.show_t1_map_with_overlay()
            else:
                self.update_status("Reset visualization settings")
                
        except Exception as e:
            self.update_status("Error resetting visualization")
            show_error_dialog("Visualization Error", "Error resetting visualization", e)

# Correction Methods and UI Interaction Handlers - Add these to the LeanT1MappingApp class

    def on_correction_changed(self, *args):
        """Called when any correction setting is changed"""
        # Update custom correction string if needed
        if self.correction_var.get().startswith('custom:'):
            self.apply_custom_correction()
        
        # Automatically recalculate T1 if we already have ROI data
        self.auto_recalculate_t1()

    def apply_custom_correction(self):
        """Apply custom correction based on checkbox selections - UNIFIED IMPLEMENTATION"""
        try:
            # Build custom correction string
            correction_str = "custom:"
            corrections_applied = []
            
            if self.apply_ll_var.get():
                correction_str += "L"
                corrections_applied.append("Look-Locker")
            if self.apply_t2star_var.get():
                correction_str += "T"
                corrections_applied.append("T2*")
            if self.apply_sat_var.get():
                correction_str += "S"
                corrections_applied.append("Saturation")
            if self.apply_inv_var.get():
                correction_str += "B"
                corrections_applied.append("Inversion")
            
            # Set the correction variable
            self.correction_var.set(correction_str)
            
            # Create description for user message
            if corrections_applied:
                correction_desc = f"Custom: {', '.join(corrections_applied)}"
            else:
                correction_desc = "Custom: No corrections selected"
            
            # Update the user
            self.update_status(f"Applied {correction_desc}")
            
            # Automatically recalculate T1 if we have ROI data
            self.auto_recalculate_t1()
            
        except Exception as e:
            self.update_status("Error applying custom correction")
            show_error_dialog("Custom Correction Error", "Error applying custom correction", e)

    def toggle_corrected_fit(self):
        """Toggle display of corrected fit curve"""
        self.show_corrected_fit = self.show_corrected_var.get()
        # The curve will update on the next mouse move - no need to redraw yet
        self.update_status(f"{'Showing' if self.show_corrected_fit else 'Hiding'} corrected fit curve")

    def auto_recalculate_t1(self, *args):
        """Recalculate T1 with existing ROI if available"""
        if not hasattr(self, 'roi_coords') or self.roi_coords is None:
            # No ROI selected yet, nothing to do
            return
                
        if not hasattr(self, 'corrected_stack') or self.corrected_stack is None:
            # No data loaded yet
            return
        
        try:
            self.update_status("Recalculating T1 with updated correction settings...")
            
            # Extract ROI data
            x1, x2, y1, y2 = self.roi_coords
            roi_data = self.corrected_stack[:, y1:y2, x1:x2]
            
            # Calculate mean signal across ROI
            mean_signal = np.mean(roi_data, axis=(1,2))
            
            # Get parameters for corrections
            flip_angle = self.sequence_params.get('FlipAngle')
            tr = self.sequence_params.get('TR')
            te = self.sequence_params.get('TE')
            
            # Get T2* from UI
            try:
                t2star = float(self.t2star_var.get())
            except:
                t2star = 40  # Default for myocardium
            
            # Calculate inversion efficiency
            inversion_efficiency = 1.0
            if flip_angle is not None:
                inversion_efficiency = calculate_inversion_efficiency(flip_angle)
            
            # Get T1 for the ROI average with updated correction settings
            try:
                # Read the cardiac correction setting from the checkbox
                apply_cardiac = self.apply_cardiac_var.get()
                
                # Modified physically_constrained_t1_fit call to include cardiac correction
                popt, self.processed_signal, self.fitted_curve, fit_status = physically_constrained_t1_fit(
                    mean_signal, self.tis, 
                    flip_angle=flip_angle,
                    tr=tr,
                    te=te,
                    t2star=t2star,
                    inversion_efficiency=inversion_efficiency,
                    timeout=1.0,
                    model=self.model_var.get(),
                    correction_method=self.correction_var.get(),
                    apply_cardiac_correction=apply_cardiac
                )
                
                a, b, self.t1_value = popt
                
            except Exception as e:
                print(f"Error in robust fitting: {e}")
                # Fall back to default values if all fitting fails
                a = np.max(mean_signal)
                b = np.ptp(mean_signal)
                self.t1_value = 1000.0
                self.processed_signal = mean_signal
                self.fitted_curve = ir_model_simple(self.tis, a, b, self.t1_value)
                fit_status = {'fallback_used': True, 'status': 'error'}
            
            # Initialize t1_map with the same shape as the ROI
            roi_height, roi_width = roi_data.shape[1:]
            self.t1_map = np.zeros((roi_height, roi_width)) * np.nan
            
            # Show progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("T1 Recalculation")
            progress_window.geometry("400x150")
            
            ttk.Label(progress_window, text="Recalculating T1 map...", 
                     font=("Arial", 12)).pack(pady=10)
            
            progress = ttk.Progressbar(progress_window, length=300, mode="determinate")
            progress.pack(pady=10)
            progress["maximum"] = roi_height
            
            self.pixel_cache = {}  # Clear pixel cache
            
            # Calculate T1 for each pixel with new settings
            for y in range(roi_height):
                for x in range(roi_width):
                    # Extract signal for this pixel
                    pixel_signal = roi_data[:, y, x].copy()
                    
                    # Skip pixels with low signal
                    if np.max(pixel_signal) < 1e-6 or np.allclose(pixel_signal, 0):
                        self.t1_map[y, x] = np.nan
                    else:
                        try:
                            # Include cardiac correction in pixel-wise calculation
                            popt, processed_signal, fitted_curve, fit_status = physically_constrained_t1_fit(
                                pixel_signal, self.tis,
                                flip_angle=flip_angle,
                                tr=tr,
                                te=te,
                                t2star=t2star,
                                inversion_efficiency=inversion_efficiency,
                                timeout=0.3,
                                model=self.model_var.get(),
                                correction_method=self.correction_var.get(),
                                apply_cardiac_correction=apply_cardiac,
                                apply_mt_correction=self.apply_mt_var.get()
                            )
                            a, b, t1 = popt
                            
                            # Store T1 value
                            self.t1_map[y, x] = t1
                            
                            # Cache pixel data for later use
                            correction_factor = fit_status.get('correction_factors', {}).get('total', 1.0)
                            t1_apparent = t1 / correction_factor
                            
                            self.pixel_cache[(x, y)] = {
                                'a': a,
                                'b': b,
                                't1': t1,
                                't1_apparent': t1_apparent,
                                'signal': processed_signal,
                                'fit': fitted_curve,
                                'model': self.model_var.get(),
                                'correction': self.correction_var.get(),
                                'r_squared': fit_status.get('r_squared'),
                                'rmse': fit_status.get('rmse'),
                                'fallback_used': fit_status.get('fallback_used', False),
                                'inversion_efficiency': inversion_efficiency,
                                'correction_factors': fit_status.get('correction_factors', {'total': 1.0})
                            }
                        
                        except Exception as e:
                            print(f"Error in pixel fit at ({x}, {y}): {e}")
                            self.t1_map[y, x] = np.nan
                
                # Update progress
                progress["value"] = y + 1
                progress_window.update()
            
            # Close progress window
            progress_window.destroy()
            
            # Calculate statistics
            valid_t1s = self.t1_map[~np.isnan(self.t1_map)]
            if len(valid_t1s) > 0:
                t1_mean = np.mean(valid_t1s)
                t1_median = np.median(valid_t1s)
                t1_std = np.std(valid_t1s)
                success_rate = len(valid_t1s) / (roi_width * roi_height) * 100
                
                # Include note about cardiac correction if applied
                cardiac_note = " (with cardiac correction)" if apply_cardiac else ""
                
                self.t1_result_var.set(f"T1 = {self.t1_value:.1f} ms (Mean ROI){cardiac_note}\n"
                                      f"Median: {t1_median:.1f} ms\n"
                                      f"StdDev: {t1_std:.1f} ms\n"
                                      f"Success: {success_rate:.1f}%")
            
            # Set T1 range if needed
            if not hasattr(self, 't1_min') or self.t1_min is None:
                if len(valid_t1s) > 0:
                    self.t1_min = np.percentile(valid_t1s, 1)
                    self.t1_max = np.percentile(valid_t1s, 99)
                    self.t1_min_var.set(f"{self.t1_min:.1f}")
                    self.t1_max_var.set(f"{self.t1_max:.1f}")
            
            # Show T1 map
            self.show_t1_map_with_overlay()
            
            # Enable export buttons if not already
            self.save_image_button.configure(state=tk.NORMAL)
            self.save_data_button.configure(state=tk.DISABLED)  # Will enable when save_t1_data is added
            
            # Update status
            cardiac_note = " (with cardiac correction)" if apply_cardiac else ""
            self.update_status(f"T1 recalculated: {self.t1_value:.1f} ms (Mean ROI){cardiac_note}")
            
        except Exception as e:
            self.update_status("Error recalculating T1")
            show_error_dialog("T1 Recalculation Error", "Error recalculating T1", e)

    def apply_enhanced_mt_correction(self):
        """Apply enhanced MT correction to the entire T1 map with pixel-wise analysis"""
        if not hasattr(self, 't1_map') or self.t1_map is None:
            messagebox.showinfo("Info", "Calculate T1 first")
            return
        
        try:
            self.update_status("Applying enhanced MT correction...")
            
            # Identify septal region
            septal_mask = identify_septal_region(self.t1_map, percentile_threshold=15)
            
            # Get ROI data
            x1, x2, y1, y2 = self.roi_coords
            roi_stack = self.corrected_stack[:, y1:y2, x1:x2]
            
            # Analyze MT effects with enhanced detection - explicit parameters
            correction_map, confidence_map, mt_stats = analyze_ti_signals_for_mt(
                self.tis, roi_stack, mask=None, enhanced_detection=True
            )
            
            # Store original T1 map if not already stored
            if not hasattr(self, 't1_map_before_mt') or self.t1_map_before_mt is None:
                self.t1_map_before_mt = self.t1_map.copy()
            
            # Apply correction with explicit parameters
            self.t1_map_mt_corrected = apply_mt_correction_to_t1map(
                self.t1_map, 
                mt_map=correction_map,
                uniform_correction=None,
                septal_mask=septal_mask,
                septal_boost=1.15  # Additional 15% for septum
            )
            
            # Update display
            self.t1_map = self.t1_map_mt_corrected.copy()
            
            # Show statistics
            mean_correction = mt_stats['mean_correction_factor']
            detected_fraction = mt_stats['detected_fraction'] * 100
            
            messagebox.showinfo("MT Correction Applied",
                              f"Enhanced MT correction applied:\n"
                              f"Mean correction factor: {mean_correction:.2f}\n"
                              f"MT effects detected in: {detected_fraction:.1f}% of pixels\n"
                              f"Septal region received additional correction")
            
            # Update visualization
            self.show_t1_map_with_overlay()
            self.update_status("Enhanced MT correction complete")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_status("Error in MT correction")
            show_error_dialog("MT Correction Error", "Error applying enhanced MT correction", e)

    def update_visualization(self, *args):
        """Update the visualization based on current settings with stable colorbar"""
        if not hasattr(self, 'main_ax') or self.t1_map is None:
            return
        
        # Store current figure dimensions and layout
        orig_figsize = self.fig.get_size_inches()
        orig_position = self.main_ax.get_position()
        
        # Store colorbar position and dimensions if it exists
        colorbar_position = None
        colorbar_size = None
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                colorbar_position = self.colorbar.ax.get_position()
                colorbar_size = self.colorbar.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted()).size
            except:
                pass
        
        # Get current settings
        self.colormap_name = self.colormap_var.get()
        self.overlay_alpha = self.alpha_var.get()
        self.show_overlay = self.overlay_var.get()
        
        # First, remove any existing overlay
        if hasattr(self, 'overlay_img') and self.overlay_img:
            try:
                self.overlay_img.remove()
            except:
                pass
            self.overlay_img = None
        
        # Remove colorbar if it exists
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                self.colorbar.remove()
            except:
                pass
            # Set to None after removing
            self.colorbar = None
        if hasattr(self, 'show_mt_button'):
            self.show_mt_button.configure(state=tk.NORMAL)
            
        # Then create new overlay if enabled
        if self.show_overlay:
            # Apply colormap directly to the T1 map
            t1_rgba = apply_colormap(
                self.t1_map, 
                colormap_name=self.colormap_name,
                min_val=self.t1_min,
                max_val=self.t1_max,
                alpha=self.overlay_alpha
            )
            
            # Display overlay
            self.overlay_img = self.main_ax.imshow(t1_rgba)
            
            # Create a new colorbar with consistent spacing from the main axis
            cax = self.fig.add_axes([0.92, 0.3, 0.02, 0.4]) if colorbar_position is None else None
            self.colorbar = self.fig.colorbar(self.overlay_img, ax=self.main_ax if cax is None else None, cax=cax)
            self.colorbar.set_label('T1 (ms)')
            
            # If we had a previous colorbar position, restore it precisely
            if colorbar_position is not None:
                self.colorbar.ax.set_position(colorbar_position)
                # Force the colorbar to maintain its exact size
                if colorbar_size is not None:
                    self.fig.canvas.draw()  # Necessary to ensure the colorbar is properly initialized
                    curr_pos = self.colorbar.ax.get_position()
                    self.colorbar.ax.set_position([curr_pos.x0, curr_pos.y0, 
                                                 colorbar_size[0], colorbar_size[1]])
        
        # Restore original figure size and axis position
        self.fig.set_size_inches(orig_figsize)
        self.main_ax.set_position(orig_position)
        
        # Make sure tight_layout doesn't override our carefully positioned elements
        try:
            self.fig.set_tight_layout(False)
        except:
            pass
        
        # Redraw canvas completely
        try:
            self.fig.canvas.draw()
        except:
            # If redrawing fails, try just updating
            self.fig.canvas.draw_idle()

# Mouse Interaction Handler and B1 Map Methods - Add these to the LeanT1MappingApp class

    def on_mouse_move(self, event):
        """Handle mouse movement over the image with improved fit display and quality metrics"""
        if event.inaxes != self.main_ax or self.t1_map is None:
            return
        
        # Get mouse position in image coordinates (relative to ROI)
        roi_x, roi_y = int(round(event.xdata)), int(round(event.ydata))
        
        # Get ROI dimensions
        roi_height, roi_width = self.t1_map.shape
        
        # Check if inside the ROI boundaries
        if 0 <= roi_x < roi_width and 0 <= roi_y < roi_height:
            # Get the actual coordinates in the original image
            x1, _, y1, _ = self.roi_coords
            x, y = roi_x + x1, roi_y + y1
            
            # Update cursor position
            self.cursor_point.set_data([roi_x], [roi_y])
            
            # Get T1 value
            try:
                t1_value = self.t1_map[roi_y, roi_x]
                if not np.isnan(t1_value):
                    # Show the T1 value in ms
                    self.cursor_info.set_text(f"ROI Pos: ({roi_x}, {roi_y})\nImg Pos: ({x}, {y})\nT1: {t1_value:.1f} ms")
                    
                    # Completely reset the curve axis
                    self.curve_ax.clear()
                    self.curve_ax.set_title("Pixel T1 Fit")
                    self.curve_ax.set_xlabel("Inversion Time (ms)")
                    self.curve_ax.set_ylabel("Signal")
                    self.curve_ax.grid(True, alpha=0.3)
                    
                    # Pixel fit calculations
                    pixel_key = (roi_x, roi_y)
                    if pixel_key in self.pixel_cache:
                        # Use the cached data
                        pixel_data = self.pixel_cache[pixel_key]
                        
                        # Get parameters from pixel data
                        a = pixel_data['a']
                        b = pixel_data['b']
                        t1_corrected = pixel_data['t1']
                        model_type = pixel_data.get('model', 'simple')
                        inversion_efficiency = pixel_data.get('inversion_efficiency', 1.0)
                        
                        # Get signal and fit from cache or recalculate if needed
                        if 'signal' in pixel_data and 'fit' in pixel_data:
                            # Use the stored signal and fit
                            signal = pixel_data['signal']
                            fitted_curve = pixel_data['fit']
                        else:
                            # Extract the pixel signal from the image
                            pixel_signal = self.corrected_stack[:, y, x].copy()
                            signal = pixel_signal
                            # Generate fit using the parameters
                            if model_type == 'classic':
                                fitted_curve = ir_model_classic(self.tis, a, b, t1_corrected, efficiency=inversion_efficiency)
                            else:
                                fitted_curve = ir_model_simple(self.tis, a, b, t1_corrected, efficiency=inversion_efficiency)
                        
                        # Get the correction factor
                        correction_factor = pixel_data.get('correction_factors', {}).get('total', 1.0)
                        # Ensure correction factor is within reasonable bounds (0.1 to 3.0)
                        correction_factor = max(0.1, min(3.0, correction_factor))
                        
                        # Calculate apparent T1 (uncorrected) from corrected T1
                        if 't1_apparent' in pixel_data:
                            t1_apparent = pixel_data['t1_apparent']
                        else:
                            t1_apparent = t1_corrected / correction_factor
                        
                        # Generate a curve that spans just beyond the data points
                        ti_min = max(0, min(self.tis) * 0.8)  # Start near zero or slightly below min TI
                        ti_max = max(self.tis) * 1.2  # Extend slightly beyond max TI
                        ti_smooth = np.linspace(ti_min, ti_max, 200)  # Enough points for smooth curve
                        
                        # Plot data points
                        self.curve_ax.scatter(self.tis, signal, color='blue', marker='o', s=40, label='Data')
                        
                        # Generate the apparent fit curve that matches data points
                        if model_type == 'classic':
                            fitted_apparent = ir_model_classic(ti_smooth, a, b, t1_apparent, efficiency=inversion_efficiency)
                        else:
                            fitted_apparent = ir_model_simple(ti_smooth, a, b, t1_apparent, efficiency=inversion_efficiency)
                        
                        # Plot the apparent fit that matches the data points
                        self.curve_ax.plot(ti_smooth, fitted_apparent, 'r-', linewidth=2, label='Fit (matches data)')
                        
                        # Optionally show the corrected fit
                        show_corrected_fit = getattr(self, 'show_corrected_fit', False)
                        if show_corrected_fit:
                            # Generate the corrected fit using corrected T1
                            if model_type == 'classic':
                                fitted_corrected = ir_model_classic(ti_smooth, a, b, t1_corrected, efficiency=inversion_efficiency)
                            else:
                                fitted_corrected = ir_model_simple(ti_smooth, a, b, t1_corrected, efficiency=inversion_efficiency)
                            
                            # Plot corrected fit
                            self.curve_ax.plot(ti_smooth, fitted_corrected, 'g--', linewidth=2, label='Fit (corrected)')
                        
                        # Set x-axis limits to ensure proper display
                        self.curve_ax.set_xlim(ti_min, ti_max)
                        
                        # Add goodness-of-fit information
                        r_squared = pixel_data.get('r_squared', None)
                        fit_quality_text = ""
                        
                        if r_squared is not None:
                            fit_quality_text = f"R² = {r_squared:.3f}\n"
                            # Add visual indicator of fit quality
                            if r_squared > 0.95:
                                fit_quality_text += "Fit Quality: Excellent"
                            elif r_squared > 0.85:
                                fit_quality_text += "Fit Quality: Good"
                            elif r_squared > 0.7:
                                fit_quality_text += "Fit Quality: Fair"
                            else:
                                fit_quality_text += "Fit Quality: Poor"
                        
                        # Add info text with both T1 values in ms
                        fit_text = f"A = {a:.2f}\nB = {b:.2f}\n"
                        fit_text += f"T1 apparent = {t1_apparent:.1f} ms\n"
                        fit_text += f"T1 corrected = {t1_corrected:.1f} ms\n"
                        fit_text += f"Correction factor = {correction_factor:.2f}\n"
                        fit_text += f"{fit_quality_text}\n"
                        fit_text += f"Model: {pixel_data.get('model', 'simple')}\n"
                        fit_text += f"Correction: {pixel_data.get('correction', 'none')}"
                        
                        if pixel_data.get('fallback_used', False):
                            fit_text += "\n(Used fallback method)"
                        
                        self.curve_ax.text(0.05, 0.95, fit_text, 
                                          transform=self.curve_ax.transAxes,
                                          va='top', ha='left',
                                          bbox=dict(facecolor='white', alpha=0.8))
                        
                        # Add legend
                        self.curve_ax.legend(loc='lower right')
                        
                        # Adjust y-axis limits to make fit more visible
                        data_range = np.ptp(signal)
                        y_padding = data_range * 0.1  # 10% padding
                        y_min = np.min(signal) - y_padding
                        y_max = np.max(signal) + y_padding
                        self.curve_ax.set_ylim(y_min, y_max)
                        
                    else:
                        # If not in cache, show message
                        self.curve_ax.text(0.5, 0.5, "Pixel data not in cache",
                                          transform=self.curve_ax.transAxes,
                                          va='center', ha='center',
                                          fontsize=12, color='orange')
                    
                    # Make sure no_data_text is hidden
                    if hasattr(self, 'no_data_text'):
                        self.no_data_text.set_visible(False)
                        
                else:
                    # No valid T1 for this pixel
                    self.cursor_info.set_text(f"ROI Pos: ({roi_x}, {roi_y})\nImg Pos: ({x}, {y})\nT1: No valid fit")
                    
                    # Clear the plot
                    self.curve_ax.clear()
                    self.curve_ax.set_title("Pixel T1 Fit")
                    self.curve_ax.set_xlabel("Inversion Time (ms)")
                    self.curve_ax.set_ylabel("Signal")
                    self.curve_ax.grid(True, alpha=0.3)
                    self.curve_ax.text(0.5, 0.5, "No valid T1 fit for this pixel",
                                      transform=self.curve_ax.transAxes,
                                      va='center', ha='center',
                                      fontsize=12, color='gray')
                    
            except (IndexError, KeyError) as e:
                # Handle indexing errors
                print(f"Index error in mouse move: {e}")
                self.cursor_info.set_text(f"ROI Pos: ({roi_x}, {roi_y})")
                
                # Clear the plot
                self.curve_ax.clear()
                self.curve_ax.set_title("Pixel T1 Fit")
                self.curve_ax.set_xlabel("Inversion Time (ms)")
                self.curve_ax.set_ylabel("Signal")
                self.curve_ax.grid(True, alpha=0.3)
                self.curve_ax.text(0.5, 0.5, "Position outside valid area",
                                  transform=self.curve_ax.transAxes,
                                  va='center', ha='center',
                                  fontsize=12, color='gray')
        else:
            # Outside ROI boundaries
            self.cursor_point.set_data([], [])
            self.cursor_info.set_text("")
            
            # Clear the plot
            self.curve_ax.clear()
            self.curve_ax.set_title("Pixel T1 Fit")
            self.curve_ax.set_xlabel("Inversion Time (ms)")
            self.curve_ax.set_ylabel("Signal")
            self.curve_ax.grid(True, alpha=0.3)
            self.curve_ax.text(0.5, 0.5, "No data selected",
                              transform=self.curve_ax.transAxes,
                              va='center', ha='center',
                              fontsize=12, color='gray')
        
        # Redraw canvas
        self.canvas.draw_idle()

    def load_b1_map(self):
        """Load a B1 map file - FIXED VERSION"""
        try:
            # Create file dialog for selecting B1 map file
            b1_file = filedialog.askopenfilename(
                title="Select B1 Map DICOM File",
                filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
            )
            
            if not b1_file:
                self.update_status("B1 map loading cancelled")
                return
            
            self.update_status(f"Loading B1 map from {b1_file}...")
            
            # Load the B1 map
            reference_image = getattr(self, 'avg_image', None)
            self.b1_map = load_and_process_b1_map(b1_file, reference_image)
            
            # FIXED: Ensure b1_map is a proper array copy
            if hasattr(self.b1_map, 'copy'):
                self.b1_map = self.b1_map.copy()
            
            # Update UI
            self.b1_info_var.set(f"B1 map loaded\n"
                               f"Size: {self.b1_map.shape[0]}×{self.b1_map.shape[1]} px\n"
                               f"Range: {np.min(self.b1_map):.2f}-{np.max(self.b1_map):.2f}\n"
                               f"Mean: {np.mean(self.b1_map):.2f}")
            
            # Enable B1 correction button if T1 map exists
            if hasattr(self, 't1_map') and self.t1_map is not None:
                self.apply_b1_button.configure(state=tk.NORMAL)
            
            # Enable coregistration button if it exists
            if hasattr(self, 'coregister_b1_button'):
                self.coregister_b1_button.configure(state=tk.NORMAL)
            
            self.update_status("B1 map loaded successfully")

            # Enable save B1 button
            if hasattr(self, 'save_b1_button'):
                self.save_b1_button.configure(state=tk.NORMAL)
            
            # Show B1 map if requested
            if hasattr(self, 'show_b1_var') and self.show_b1_var.get():
                self.display_b1_map()
            
        except Exception as e:
            self.update_status("Error loading B1 map")
            show_error_dialog("B1 Map Loading Error", "Error loading B1 map file", e)

    def coregister_b1_map(self):
        """Coregister B1 map to match T1 map dimensions"""
        if not hasattr(self, 'b1_map') or self.b1_map is None:
            messagebox.showinfo("Info", "Please load a B1 map first")
            return
        
        # We need a reference image to register to
        reference_image = None
        
        # First choice: use the T1 map if available
        if hasattr(self, 't1_map') and self.t1_map is not None:
            # Get the full image, not just the ROI
            reference_image = self.avg_image
            self.update_status("Coregistering B1 map to match T1 map dimensions...")
            
        # Second choice: use the average image if available
        elif hasattr(self, 'avg_image') and self.avg_image is not None:
            reference_image = self.avg_image
            self.update_status("Coregistering B1 map to match average image dimensions...")
            
        # Third choice: use the first image from the stack if available
        elif hasattr(self, 'corrected_stack') and self.corrected_stack is not None:
            reference_image = self.corrected_stack[0]
            self.update_status("Coregistering B1 map to match image stack dimensions...")
        
        else:
            messagebox.showinfo("Info", "No reference image available for coregistration")
            return
        
        try:
            # Get the registration method
            reg_method = self.reg_method_var.get()
            
            # Show progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("B1 Map Coregistration")
            progress_window.geometry("400x150")
            
            ttk.Label(progress_window, text=f"Coregistering B1 map using {reg_method} method...", 
                     font=("Arial", 12)).pack(pady=10)
            
            progress = ttk.Progressbar(progress_window, length=300, mode="indeterminate")
            progress.pack(pady=10)
            progress.start()
            progress_window.update()
            
            # Store the original B1 map for reference
            if not hasattr(self, 'original_b1_map'):
                self.original_b1_map = self.b1_map.copy()
            
            # Perform the coregistration
            self.b1_map = coregister_b1_to_reference(
                self.original_b1_map, 
                reference_image,
                method=reg_method,
                visualize=False  # We'll show our own visualization
            )
            
            # Update UI
            self.b1_info_var.set(f"B1 map coregistered\n"
                               f"Size: {self.b1_map.shape[0]}×{self.b1_map.shape[1]} px\n"
                               f"Range: {np.min(self.b1_map):.2f}-{np.max(self.b1_map):.2f}\n"
                               f"Mean: {np.mean(self.b1_map):.2f}")
            
            # Enable B1 correction button if T1 map exists
            if hasattr(self, 't1_map') and self.t1_map is not None:
                self.apply_b1_button.configure(state=tk.NORMAL)
            
            # Close progress window
            progress.stop()
            progress_window.destroy()
            
            # Show coregistration results
            self.show_coregistration_results()
            
            self.update_status("B1 map coregistered successfully")
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            self.update_status("Error in B1 map coregistration")
            show_error_dialog("Coregistration Error", "Error coregistering B1 map", e)

    def apply_b1_correction(self):
        """Apply B1 correction to the current T1 map - FIXED ARRAY HANDLING"""
        if not hasattr(self, 'b1_map') or self.b1_map is None:
            messagebox.showinfo("Info", "Please load a B1 map first")
            return
        
        if not hasattr(self, 't1_map') or self.t1_map is None:
            messagebox.showinfo("Info", "Please calculate T1 first")
            return
        
        try:
            self.update_status("Applying B1 correction...")
            
            # Check if dimensions match
            if self.b1_map.shape != self.avg_image.shape:
                # Ask if user wants to coregister
                response = messagebox.askyesno(
                    "Dimension Mismatch", 
                    "B1 map dimensions don't match reference image.\n"
                    "Would you like to coregister the B1 map first?"
                )
                
                if response:
                    # Run coregistration
                    self.coregister_b1_map()
                    
                    # Check again after coregistration
                    if self.b1_map.shape != self.avg_image.shape:
                        messagebox.showwarning(
                            "Warning", 
                            "B1 map dimensions still don't match.\n"
                            "Will attempt to extract matching ROI or resample."
                        )
            
            # Extract ROI coordinates
            x1, x2, y1, y2 = self.roi_coords
            
            # FIXED: Handle B1 map extraction with explicit bounds checking
            if (0 <= y1 < y2 <= self.b1_map.shape[0] and 
                0 <= x1 < x2 <= self.b1_map.shape[1]):
                # B1 map covers the ROI, extract matching region
                b1_roi = self.b1_map[y1:y2, x1:x2].copy()  # Explicit copy
                
            else:
                # Need to resample - handle various cases
                from scipy.interpolate import RegularGridInterpolator
                
                try:
                    # Create normalized coordinates for both images
                    b1_y_norm = np.linspace(0, 1, self.b1_map.shape[0])
                    b1_x_norm = np.linspace(0, 1, self.b1_map.shape[1])
                    
                    # Create ROI normalized coordinates
                    roi_y_start = y1 / self.avg_image.shape[0]
                    roi_y_end = y2 / self.avg_image.shape[0]
                    roi_x_start = x1 / self.avg_image.shape[1]
                    roi_x_end = x2 / self.avg_image.shape[1]
                    
                    # Create interpolator for B1 map
                    b1_interp = RegularGridInterpolator(
                        (b1_y_norm, b1_x_norm), 
                        self.b1_map,
                        bounds_error=False,
                        fill_value=1.0  # Default to nominal B1 outside bounds
                    )
                    
                    # Create coordinates for resampling
                    roi_height, roi_width = self.t1_map.shape
                    y_roi_norm = np.linspace(roi_y_start, roi_y_end, roi_height)
                    x_roi_norm = np.linspace(roi_x_start, roi_x_end, roi_width)
                    
                    # Create meshgrid of coordinates
                    Y, X = np.meshgrid(y_roi_norm, x_roi_norm, indexing='ij')
                    points = np.column_stack((Y.flatten(), X.flatten()))
                    
                    # Interpolate
                    b1_roi = b1_interp(points).reshape(roi_height, roi_width)
                    
                    # Notify user about resampling
                    messagebox.showinfo(
                        "B1 Map Resampling", 
                        "B1 map has been resampled to match the ROI dimensions."
                    )
                    
                except Exception as interp_error:
                    # Fallback to uniform B1 map
                    messagebox.showwarning(
                        "B1 Resampling Failed",
                        f"Failed to resample B1 map: {str(interp_error)}\n"
                        "Using uniform B1 map (no correction)."
                    )
                    b1_roi = np.ones_like(self.t1_map)
            
            # FIXED: Apply correction with explicit copies to avoid view issues
            t1_map_copy = self.t1_map.copy()
            b1_roi_copy = b1_roi.copy()
            
            t1_corrected, correction_map = apply_b1_correction_to_t1_map(
                t1_map_copy, b1_roi_copy, 
                flip_angle=self.sequence_params.get('FlipAngle'), 
                tr=self.sequence_params.get('TR')
            )
            
            # Store the original T1 map if not already stored
            if not hasattr(self, 't1_map_original'):
                self.t1_map_original = self.t1_map.copy()
            
            # FIXED: Update T1 map with explicit copy to avoid view issues
            self.t1_map = t1_corrected.copy()
            self.b1_correction_map = correction_map.copy()
            
            # Calculate new statistics with proper copy
            valid_t1s = self.t1_map[~np.isnan(self.t1_map)]
            if len(valid_t1s) > 0:
                t1_mean = np.mean(valid_t1s)
                t1_median = np.median(valid_t1s)
                t1_std = np.std(valid_t1s)
                
                # Update the mean ROI T1 value using the correction factor
                valid_corrections = correction_map[~np.isnan(correction_map)]
                if len(valid_corrections) > 0:
                    mean_correction = np.median(valid_corrections)
                    self.t1_value = self.t1_value * mean_correction
                
                self.t1_result_var.set(f"T1 = {self.t1_value:.1f} ms (B1 corrected)\n"
                                     f"Median: {t1_median:.1f} ms\n"
                                     f"StdDev: {t1_std:.1f} ms\n"
                                     f"B1 correction applied")
            
            # Update visualization
            self.update_visualization()
            
            self.update_status("B1 correction applied successfully")
            
        except Exception as e:
            self.update_status("Error applying B1 correction")
            show_error_dialog("B1 Correction Error", "Error applying B1 correction", e)

    def toggle_b1_display(self):
        """Toggle display of B1 map with better error handling"""
        try:
            if self.show_b1_var.get():
                self.display_b1_map()
            else:
                # Return to T1 map display
                if hasattr(self, 't1_map') and self.t1_map is not None:
                    self.show_t1_map_with_overlay()
                else:
                    # No T1 map, just show the regular image
                    self.show_stack_overview()
        except Exception as e:
            print(f"Error in toggle_b1_display: {e}")
            messagebox.showwarning("Display Error", f"Error displaying B1 map: {str(e)}")
            # Reset the toggle to unchecked
            self.show_b1_var.set(False)

    def display_b1_map(self):
        """Display the B1 map with better error handling"""
        if not hasattr(self, 'b1_map') or self.b1_map is None:
            messagebox.showinfo("Info", "No B1 map loaded")
            return
        
        # Clear the figure
        self.fig.clear()
        
        # Create subplot for B1 map
        self.main_ax = self.fig.add_subplot(111)
        
        # Create a masked version of the B1 map
        masked_b1 = np.ma.masked_invalid(self.b1_map)
        
        # Apply a colormap designed for B1 visualization
        im = self.main_ax.imshow(masked_b1, cmap='coolwarm', vmin=0.5, vmax=1.5)
        
        # Add colorbar
        self.colorbar = self.fig.colorbar(im, ax=self.main_ax)
        self.colorbar.set_label('B1 Factor (1.0 = Nominal)')
        
        # Add title with B1 statistics (with error handling)
        valid_b1 = masked_b1[~masked_b1.mask]
        if len(valid_b1) > 0:
            mean_b1 = np.mean(valid_b1)
            median_b1 = np.median(valid_b1)
            std_b1 = np.std(valid_b1)
            self.main_ax.set_title(f"B1 Map (Mean: {mean_b1:.2f}, Median: {median_b1:.2f}, StdDev: {std_b1:.2f})")
        else:
            self.main_ax.set_title("B1 Map (No valid values)")
        
        # Remove axis labels
        self.main_ax.axis('off')
        
        # Add ROI indicator if available
        if hasattr(self, 'roi_coords') and self.roi_coords is not None:
            try:
                x1, x2, y1, y2 = self.roi_coords
                from matplotlib.patches import Rectangle
                rect = Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    edgecolor='lime', facecolor='none', linewidth=2
                )
                self.main_ax.add_patch(rect)
            except (ValueError, TypeError) as e:
                # If ROI coordinates are invalid, just skip drawing the rectangle
                print(f"Error drawing ROI rectangle: {e}")
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.update_status("Displaying B1 map")

    def show_coregistration_results(self):
        """Show before/after results of B1 map coregistration"""
        if not hasattr(self, 'original_b1_map') or not hasattr(self, 'b1_map'):
            return
        
        # Create a new window
        result_window = tk.Toplevel(self.root)
        result_window.title("B1 Map Coregistration Results")
        result_window.geometry("1000x600")
        
        # Create figure
        fig = plt.Figure(figsize=(10, 6))
        
        # Add original B1 map
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(self.original_b1_map, cmap='viridis', vmin=0.5, vmax=1.5)
        ax1.set_title("Original B1 Map")
        ax1.axis('off')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Add coregistered B1 map
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(self.b1_map, cmap='viridis', vmin=0.5, vmax=1.5)
        ax2.set_title("Coregistered B1 Map")
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Add reference image with B1 overlay
        ax3 = fig.add_subplot(133)
        
        # Choose appropriate reference image
        if hasattr(self, 'avg_image') and self.avg_image is not None:
            reference = self.avg_image
            title = "Average Image with B1 Overlay"
        elif hasattr(self, 'corrected_stack') and self.corrected_stack is not None:
            reference = self.corrected_stack[0]
            title = "Reference Image with B1 Overlay"
        else:
            # Create dummy reference
            reference = np.ones_like(self.b1_map)
            title = "B1 Map"
        
        # Display reference image
        ax3.imshow(reference, cmap='gray')
        
        # Create B1 overlay
        # Normalize B1 map to 0-1 range
        if np.max(self.b1_map) > np.min(self.b1_map):
            b1_norm = (self.b1_map - np.min(self.b1_map)) / (np.max(self.b1_map) - np.min(self.b1_map))
        else:
            b1_norm = np.zeros_like(self.b1_map)
        
        # Create colormap with transparency
        cmap = plt.get_cmap('jet')
        b1_rgba = cmap(b1_norm)
        b1_rgba[..., 3] = 0.5  # Set alpha for all pixels
        
        # Add overlay
        ax3.imshow(b1_rgba)
        ax3.set_title(title)
        ax3.axis('off')
        
        # Add canvas to window
        canvas = FigureCanvasTkAgg(fig, result_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add buttons
        button_frame = ttk.Frame(result_window)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Use Original B1", 
                  command=lambda: self.restore_original_b1()).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Keep Coregistered B1", 
                  command=result_window.destroy).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Save Results", 
                  command=lambda: self.save_coregistration_figure(fig)).pack(side=tk.RIGHT, padx=10)

    def restore_original_b1(self):
        """Restore the original B1 map (before coregistration)"""
        if hasattr(self, 'original_b1_map'):
            self.b1_map = self.original_b1_map.copy()
            self.b1_info_var.set(f"Restored original B1 map\n"
                               f"Size: {self.b1_map.shape[0]}×{self.b1_map.shape[1]} px\n"
                               f"Range: {np.min(self.b1_map):.2f}-{np.max(self.b1_map):.2f}\n"
                               f"Mean: {np.mean(self.b1_map):.2f}")
            self.update_status("Restored original B1 map")
            
            # Update display if showing B1 map
            if self.show_b1_var.get():
                self.display_b1_map()

    def save_coregistration_figure(self, fig):
        """Save the coregistration comparison figure"""
        file_path = filedialog.asksaveasfilename(
            title="Save Coregistration Results",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPG files", "*.jpg"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Coregistration results saved to {file_path}")

    def save_b1_map(self):
        """Save the current B1 map to a file"""
        if not hasattr(self, 'b1_map') or self.b1_map is None:
            messagebox.showinfo("Info", "No B1 map to save")
            return
        
        try:
            # Ask for file path
            file_path = filedialog.asksaveasfilename(
                title="Save B1 Map",
                defaultextension=".npy",
                filetypes=[
                    ("NumPy files", "*.npy"),
                    ("MATLAB files", "*.mat"),
                    ("CSV files", "*.csv"),
                    ("DICOM files", "*.dcm"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return
            
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.npy':
                # Save as NumPy array
                np.save(file_path, self.b1_map)
                
            elif ext == '.mat':
                # Save as MATLAB file
                try:
                    import scipy.io
                    scipy.io.savemat(file_path, {'b1_map': self.b1_map})
                except ImportError:
                    messagebox.showwarning("Missing Library", 
                                         "scipy is required to save MATLAB files")
                    return
                    
            elif ext == '.csv':
                # Save as CSV
                np.savetxt(file_path, self.b1_map, delimiter=',', fmt='%.6f')
                
            elif ext == '.dcm':
                # Save as DICOM
                self.save_b1_map_as_dicom(file_path)
                
            else:
                # Default to NumPy format
                np.save(file_path + '.npy', self.b1_map)
                file_path += '.npy'
            
            self.update_status(f"B1 map saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"B1 map saved to {file_path}")
            
        except Exception as e:
            self.update_status("Error saving B1 map")
            show_error_dialog("Save Error", "Error saving B1 map", e)


    def save_b1_map_as_dicom(self, output_path):
        """Save B1 map as a DICOM file"""
        try:
            # Create DICOM dataset
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.ImplementationClassUID = generate_uid()
            file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
            
            # Create file dataset
            ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
            
            # Set required DICOM tags
            ds.PatientName = "B1_MAP"
            ds.PatientID = "B1MAP001"
            ds.StudyInstanceUID = generate_uid()
            ds.SeriesInstanceUID = generate_uid()
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
            ds.SOPInstanceUID = generate_uid()
            ds.Modality = 'MR'
            ds.SeriesDescription = 'B1_MAP_DUAL_FLIP_ANGLE'
            ds.ImageType = ['DERIVED', 'PRIMARY', 'B1_MAP']
            
            # Convert B1 map to uint16 (scale to 0-1000 for 0.0-2.0 range)
            b1_scaled = np.clip(self.b1_map * 500, 0, 1000).astype(np.uint16)
            
            # Image data
            ds.Rows, ds.Columns = b1_scaled.shape
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            
            # Rescale to get back to B1 values
            ds.RescaleSlope = 0.002  # 1/500
            ds.RescaleIntercept = 0.0
            ds.RescaleType = 'B1'
            
            # Set pixel data
            ds.PixelData = b1_scaled.tobytes()
            
            # Window/Level
            ds.WindowCenter = "500"  # Corresponds to B1=1.0
            ds.WindowWidth = "300"   # Shows range 0.7-1.3
            
            # Save
            ds.save_as(output_path, write_like_original=False)
            
        except Exception as e:
            raise Exception(f"Error saving B1 map as DICOM: {str(e)}") 

    
    def calculate_b1_map_from_dual_flip_angle(image_fa1, image_fa2, fa1_deg, fa2_deg, tr=None, mask_threshold=0.1):
        """
        Calculate B1 map from dual flip angle images
        
        Parameters:
        -----------
        image_fa1 : ndarray
            Image acquired with first flip angle
        image_fa2 : ndarray
            Image acquired with second flip angle
        fa1_deg : float
            First flip angle in degrees
        fa2_deg : float
            Second flip angle in degrees
        tr : float, optional
            Repetition time in ms (for Ernst angle correction if needed)
        mask_threshold : float
            Threshold for creating signal mask (fraction of max signal)
            
        Returns:
        --------
        b1_map : ndarray
            B1 field map (1.0 = nominal field strength)
        confidence_map : ndarray
            Confidence map based on signal strength
        """
        import numpy as np
        from scipy.optimize import minimize_scalar
        
        # Ensure images have the same shape
        if image_fa1.shape != image_fa2.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert flip angles to radians
        fa1_rad = np.radians(fa1_deg)
        fa2_rad = np.radians(fa2_deg)
        
        # Create mask based on signal intensity
        max_signal = max(np.max(image_fa1), np.max(image_fa2))
        signal_mask = ((image_fa1 > mask_threshold * max_signal) & 
                       (image_fa2 > mask_threshold * max_signal))
        
        # Initialize B1 map
        b1_map = np.ones_like(image_fa1, dtype=np.float32)
        confidence_map = np.zeros_like(image_fa1, dtype=np.float32)
        
        # Method 1: Analytical solution for small angle approximation
        # This works well when angles are relatively small and TR >> T1
        if fa1_deg < 60 and fa2_deg < 60:
            # For small angles, sin(x) ≈ x, so S ∝ B1*FA
            # Therefore: B1 = sqrt((S1/S2) * (FA2/FA1))
            
            ratio = np.zeros_like(image_fa1, dtype=np.float32)
            ratio[signal_mask] = image_fa1[signal_mask] / image_fa2[signal_mask]
            
            # Avoid division by zero and unrealistic ratios
            valid_ratio = (ratio > 0.1) & (ratio < 10) & signal_mask
            
            b1_map[valid_ratio] = np.sqrt(ratio[valid_ratio] * (fa2_deg / fa1_deg))
            
        else:
            # Method 2: Full solution for arbitrary angles
            # S1/S2 = sin(B1*FA1) / sin(B1*FA2)
            
            # Vectorized solution using Newton's method
            # We solve: f(B1) = S1/S2 - sin(B1*FA1)/sin(B1*FA2) = 0
            
            ratio_map = np.zeros_like(image_fa1, dtype=np.float32)
            ratio_map[signal_mask] = image_fa1[signal_mask] / image_fa2[signal_mask]
            
            # For each pixel in the mask, solve for B1
            y_indices, x_indices = np.where(signal_mask)
            
            for i in range(len(y_indices)):
                y, x = y_indices[i], x_indices[i]
                ratio = ratio_map[y, x]
                
                # Skip if ratio is unrealistic
                if ratio < 0.1 or ratio > 10:
                    continue
                
                # Define the function to minimize
                def objective(b1):
                    if b1 <= 0:
                        return 1e10
                    sin1 = np.sin(b1 * fa1_rad)
                    sin2 = np.sin(b1 * fa2_rad)
                    if abs(sin2) < 1e-6:
                        return 1e10
                    return abs(ratio - sin1/sin2)
                
                # Find B1 that minimizes the objective
                result = minimize_scalar(objective, bounds=(0.1, 2.0), method='bounded')
                
                if result.success and result.fun < 0.1:  # Good fit
                    b1_map[y, x] = result.x
                    confidence_map[y, x] = 1.0 / (1.0 + result.fun)
                else:
                    # Fall back to small angle approximation
                    b1_map[y, x] = np.sqrt(ratio * (fa2_deg / fa1_deg))
                    confidence_map[y, x] = 0.5
        
        # Post-processing: smooth the B1 map to reduce noise
        from scipy.ndimage import gaussian_filter
        
        # Only smooth within the mask
        b1_smoothed = b1_map.copy()
        
        # Create a temporary map for smoothing
        temp_b1 = b1_map.copy()
        temp_b1[~signal_mask] = np.nan
        
        # Apply Gaussian smoothing
        sigma = 2.0  # Smoothing parameter
        smoothed = gaussian_filter(temp_b1, sigma=sigma)
        
        # Also smooth the mask to get weights
        mask_smoothed = gaussian_filter(signal_mask.astype(float), sigma=sigma)
        
        # Normalize by the smoothed mask
        valid_smooth = mask_smoothed > 0.1
        b1_smoothed[valid_smooth] = smoothed[valid_smooth] / mask_smoothed[valid_smooth]
        
        # Ensure B1 values are in reasonable range
        b1_smoothed = np.clip(b1_smoothed, 0.5, 1.5)
        
        # Update confidence based on signal strength
        avg_signal = (image_fa1 + image_fa2) / 2
        confidence_map[signal_mask] = np.clip(avg_signal[signal_mask] / max_signal, 0, 1)
        
        return b1_smoothed, confidence_map


    def load_dual_flip_angle_images(fa1_paths, fa2_paths):
        """
        Load two sets of DICOM images with different flip angles
        
        Parameters:
        -----------
        fa1_paths : list
            Paths to DICOM files for first flip angle
        fa2_paths : list
            Paths to DICOM files for second flip angle
            
        Returns:
        --------
        image_fa1 : ndarray
            Average image from first flip angle series
        image_fa2 : ndarray
            Average image from second flip angle series
        fa1 : float
            First flip angle in degrees
        fa2 : float
            Second flip angle in degrees
        tr : float
            Repetition time in ms
        """
        import numpy as np
        import pydicom
        
        # Load first flip angle series
        images_fa1 = []
        flip_angles_1 = []
        tr_values_1 = []
        
        for path in fa1_paths:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            
            # Apply rescale if available
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            img = img * slope + intercept
            
            images_fa1.append(img)
            
            if hasattr(ds, 'FlipAngle'):
                flip_angles_1.append(float(ds.FlipAngle))
            if hasattr(ds, 'RepetitionTime'):
                tr_values_1.append(float(ds.RepetitionTime))
        
        # Load second flip angle series
        images_fa2 = []
        flip_angles_2 = []
        tr_values_2 = []
        
        for path in fa2_paths:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            
            # Apply rescale if available
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            img = img * slope + intercept
            
            images_fa2.append(img)
            
            if hasattr(ds, 'FlipAngle'):
                flip_angles_2.append(float(ds.FlipAngle))
            if hasattr(ds, 'RepetitionTime'):
                tr_values_2.append(float(ds.RepetitionTime))
        
        # Check dimensions match
        shapes_1 = [img.shape for img in images_fa1]
        shapes_2 = [img.shape for img in images_fa2]
        
        if len(set(shapes_1)) > 1 or len(set(shapes_2)) > 1:
            raise ValueError("All images in each series must have the same dimensions")
        
        if shapes_1[0] != shapes_2[0]:
            raise ValueError("Images from both flip angle series must have the same dimensions")
        
        # Average images if multiple slices
        if len(images_fa1) > 1:
            image_fa1 = np.mean(np.stack(images_fa1, axis=0), axis=0)
        else:
            image_fa1 = images_fa1[0]
        
        if len(images_fa2) > 1:
            image_fa2 = np.mean(np.stack(images_fa2, axis=0), axis=0)
        else:
            image_fa2 = images_fa2[0]
        
        # Get flip angles
        if flip_angles_1:
            fa1 = np.mean(flip_angles_1)
        else:
            raise ValueError("No flip angle found in first series")
        
        if flip_angles_2:
            fa2 = np.mean(flip_angles_2)
        else:
            raise ValueError("No flip angle found in second series")
        
        # Ensure flip angles are different
        if abs(fa1 - fa2) < 5:
            raise ValueError(f"Flip angles too similar: {fa1}° and {fa2}°. Need at least 5° difference.")
        
        # Get TR (should be the same for both)
        tr = None
        if tr_values_1:
            tr = np.mean(tr_values_1)
        elif tr_values_2:
            tr = np.mean(tr_values_2)
        
        return image_fa1, image_fa2, fa1, fa2, tr


    def create_b1_map_from_fa_series(self):
        """
        GUI method to create B1 map from two flip angle series
        This method should be added to the LeanT1MappingApp class
        """
        try:
            # Create dialog for selecting two series
            dialog = tk.Toplevel(self.root)
            dialog.title("Create B1 Map from Dual Flip Angle")
            dialog.geometry("500x400")
            
            # Instructions
            ttk.Label(dialog, text="Select DICOM files for two different flip angle acquisitions",
                     font=("Arial", 12, "bold")).pack(pady=10)
            
            # First flip angle series
            fa1_frame = ttk.LabelFrame(dialog, text="First Flip Angle Series", padding=10)
            fa1_frame.pack(fill=tk.X, padx=20, pady=10)
            
            self.fa1_files = []
            self.fa1_info_var = tk.StringVar(value="No files selected")
            ttk.Label(fa1_frame, textvariable=self.fa1_info_var).pack()
            
            def select_fa1_files():
                files = list(filedialog.askopenfilenames(
                    title="Select First Flip Angle DICOM Files",
                    filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
                ))
                if files:
                    self.fa1_files = files
                    # Try to read flip angle from first file
                    try:
                        ds = pydicom.dcmread(files[0])
                        fa = getattr(ds, 'FlipAngle', 'Unknown')
                        self.fa1_info_var.set(f"{len(files)} files selected (FA: {fa}°)")
                    except:
                        self.fa1_info_var.set(f"{len(files)} files selected")
            
            ttk.Button(fa1_frame, text="Select Files", command=select_fa1_files).pack(pady=5)
            
            # Second flip angle series
            fa2_frame = ttk.LabelFrame(dialog, text="Second Flip Angle Series", padding=10)
            fa2_frame.pack(fill=tk.X, padx=20, pady=10)
            
            self.fa2_files = []
            self.fa2_info_var = tk.StringVar(value="No files selected")
            ttk.Label(fa2_frame, textvariable=self.fa2_info_var).pack()
            
            def select_fa2_files():
                files = list(filedialog.askopenfilenames(
                    title="Select Second Flip Angle DICOM Files",
                    filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
                ))
                if files:
                    self.fa2_files = files
                    # Try to read flip angle from first file
                    try:
                        ds = pydicom.dcmread(files[0])
                        fa = getattr(ds, 'FlipAngle', 'Unknown')
                        self.fa2_info_var.set(f"{len(files)} files selected (FA: {fa}°)")
                    except:
                        self.fa2_info_var.set(f"{len(files)} files selected")
            
            ttk.Button(fa2_frame, text="Select Files", command=select_fa2_files).pack(pady=5)
            
            # Buttons
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=20)
            
            def process_b1_map():
                if not self.fa1_files or not self.fa2_files:
                    messagebox.showwarning("Missing Files", 
                                         "Please select files for both flip angle series")
                    return
                
                try:
                    self.update_status("Loading dual flip angle images...")
                    
                    # Load the images
                    image_fa1, image_fa2, fa1, fa2, tr = load_dual_flip_angle_images(
                        self.fa1_files, self.fa2_files
                    )
                    
                    self.update_status(f"Calculating B1 map from FA {fa1:.1f}° and {fa2:.1f}°...")
                    
                    # Calculate B1 map
                    self.b1_map, confidence_map = calculate_b1_map_from_dual_flip_angle(
                        image_fa1, image_fa2, fa1, fa2, tr
                    )
                    
                    # Update UI
                    self.b1_info_var.set(f"B1 map calculated from dual FA\n"
                                       f"FA1: {fa1:.1f}°, FA2: {fa2:.1f}°\n"
                                       f"Size: {self.b1_map.shape[0]}×{self.b1_map.shape[1]} px\n"
                                       f"Range: {np.min(self.b1_map):.2f}-{np.max(self.b1_map):.2f}")
                    
                    # Enable B1 correction button if T1 map exists
                    if hasattr(self, 't1_map') and self.t1_map is not None:
                        self.apply_b1_button.configure(state=tk.NORMAL)
                    
                    # Enable coregistration button
                    if hasattr(self, 'coregister_b1_button'):
                        self.coregister_b1_button.configure(state=tk.NORMAL)
                    
                    # Show B1 map visualization
                    self.show_b1_calculation_results(image_fa1, image_fa2, 
                                                   self.b1_map, confidence_map,
                                                   fa1, fa2)
                    
                    self.update_status("B1 map calculated successfully")
                    dialog.destroy()
                    
                except Exception as e:
                    self.update_status("Error calculating B1 map")
                    show_error_dialog("B1 Calculation Error", 
                                    "Error calculating B1 map from dual flip angle", e)
            
            ttk.Button(button_frame, text="Calculate B1 Map", 
                      command=process_b1_map).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", 
                      command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            show_error_dialog("B1 Map Creation Error", 
                            "Error creating B1 map from flip angle series", e)

    def show_b1_calculation_results(self, image_fa1, image_fa2, b1_map, confidence_map, fa1, fa2):
        """
        Show the results of B1 map calculation
        This method should be added to the LeanT1MappingApp class
        """
        # Create a new window for results
        result_window = tk.Toplevel(self.root)
        result_window.title("B1 Map Calculation Results")
        result_window.geometry("1200x800")
        
        # Create figure
        fig = plt.Figure(figsize=(12, 8))
        
        # First flip angle image
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(image_fa1, cmap='gray')
        ax1.set_title(f"Flip Angle {fa1:.1f}°")
        ax1.axis('off')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Second flip angle image
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(image_fa2, cmap='gray')
        ax2.set_title(f"Flip Angle {fa2:.1f}°")
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Signal ratio
        ax3 = fig.add_subplot(2, 3, 3)
        ratio_map = np.zeros_like(image_fa1)
        mask = (image_fa2 > 0.1 * np.max(image_fa2))
        ratio_map[mask] = image_fa1[mask] / image_fa2[mask]
        im3 = ax3.imshow(ratio_map, cmap='coolwarm', vmin=0.5, vmax=1.5)
        ax3.set_title("Signal Ratio (FA1/FA2)")
        ax3.axis('off')
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # B1 map
        ax4 = fig.add_subplot(2, 3, 4)
        im4 = ax4.imshow(b1_map, cmap='viridis', vmin=0.7, vmax=1.3)
        ax4.set_title("B1 Map")
        ax4.axis('off')
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('B1 Factor')
        
        # Confidence map
        ax5 = fig.add_subplot(2, 3, 5)
        im5 = ax5.imshow(confidence_map, cmap='plasma', vmin=0, vmax=1)
        ax5.set_title("Confidence Map")
        ax5.axis('off')
        cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.set_label('Confidence')
        
        # Histogram of B1 values
        ax6 = fig.add_subplot(2, 3, 6)
        valid_b1 = b1_map[confidence_map > 0.5]
        ax6.hist(valid_b1.flatten(), bins=50, density=True, alpha=0.7, color='blue')
        ax6.axvline(1.0, color='red', linestyle='--', label='Nominal B1')
        ax6.set_xlabel('B1 Factor')
        ax6.set_ylabel('Density')
        ax6.set_title('B1 Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"B1 Statistics:\n"
        stats_text += f"Mean: {np.mean(valid_b1):.3f}\n"
        stats_text += f"Median: {np.median(valid_b1):.3f}\n"
        stats_text += f"Std Dev: {np.std(valid_b1):.3f}\n"
        stats_text += f"Range: [{np.min(valid_b1):.3f}, {np.max(valid_b1):.3f}]"
        
        ax6.text(0.98, 0.98, stats_text, transform=ax6.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        
        # Add canvas to window
        canvas = FigureCanvasTkAgg(fig, result_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add buttons
        button_frame = ttk.Frame(result_window)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_b1_results():
            file_path = filedialog.asksaveasfilename(
                title="Save B1 Map Results",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"B1 map results saved to {file_path}")
        
        ttk.Button(button_frame, text="Save Figure", 
                  command=save_b1_results).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Close", 
                  command=result_window.destroy).pack(side=tk.RIGHT, padx=5)

# Quality Assessment and Data Export Methods - Add these to the LeanT1MappingApp class

    def show_quality_maps(self):
        """Show comprehensive quality maps"""
        if not hasattr(self, 't1_map') or self.t1_map is None:
            messagebox.showinfo("Info", "Calculate T1 first")
            return
        
        if not hasattr(self, 'pixel_cache') or not self.pixel_cache:
            messagebox.showinfo("Info", "No pixel cache available")
            return
        
        try:
            self.update_status("Generating quality maps...")
            
            # Create quality maps
            quality_maps = create_comprehensive_quality_maps(self.t1_map, self.pixel_cache)
            
            # Create a new window for quality maps
            quality_window = tk.Toplevel(self.root)
            quality_window.title("T1 Mapping Quality Assessment")
            quality_window.geometry("1200x800")
            
            # Create figure with quality maps
            fig = plt.Figure(figsize=(12, 8))
            visualize_quality_maps(quality_maps, fig)
            
            # Add figure to window
            canvas = FigureCanvasTkAgg(fig, quality_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add save button
            save_frame = ttk.Frame(quality_window, padding=10)
            save_frame.pack(fill=tk.X)
            
            ttk.Button(save_frame, text="Save Quality Maps", 
                      command=lambda: self.save_quality_figure(fig)).pack(side=tk.RIGHT, padx=5)
            
            ttk.Button(save_frame, text="Close", 
                      command=quality_window.destroy).pack(side=tk.RIGHT, padx=5)
            
            self.update_status("Quality maps generated")
            
        except Exception as e:
            self.update_status("Error generating quality maps")
            show_error_dialog("Quality Assessment Error", "Error generating quality maps", e)

    def save_quality_figure(self, fig):
        """Save quality maps figure"""
        try:
            # Ask for the file path
            file_path = filedialog.asksaveasfilename(
                title="Save Quality Maps",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPG files", "*.jpg"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return
            
            # Save the figure
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            messagebox.showinfo("Success", f"Quality maps saved to {file_path}")
            
        except Exception as e:
            show_error_dialog("Save Error", "Error saving quality maps", e)

    def show_quality_report(self):
        """Generate and display quality report"""
        if not hasattr(self, 't1_map') or self.t1_map is None:
            messagebox.showinfo("Info", "Calculate T1 first")
            return
        
        if not hasattr(self, 'pixel_cache') or not self.pixel_cache:
            messagebox.showinfo("Info", "No pixel cache available")
            return
        
        try:
            self.update_status("Generating quality report...")
            
            # Generate quality report
            report = generate_quality_report(
                self.t1_map, 
                self.pixel_cache, 
                self.sequence_params, 
                self.roi_coords
            )
            
            # Create report window
            report_window = tk.Toplevel(self.root)
            report_window.title("T1 Mapping Quality Report")
            report_window.geometry("600x700")
            
            # Create notebook for tabs
            notebook = ttk.Notebook(report_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Summary tab
            summary_frame = ttk.Frame(notebook, padding=10)
            notebook.add(summary_frame, text="Summary")
            
            # Add summary content
            quality_score = report['summary'].get('quality_score', 0)
            quality_assessment = report['summary'].get('quality_assessment', 'Unknown')
            
            summary_header = ttk.Label(summary_frame, 
                                      text=f"Quality Score: {quality_score}/100 - {quality_assessment}",
                                      font=("Arial", 14, "bold"))
            summary_header.pack(pady=10)
            
            # Add quality factors
            if 'quality_factors' in report['summary']:
                factors_frame = ttk.LabelFrame(summary_frame, text="Quality Factors", padding=10)
                factors_frame.pack(fill=tk.X, pady=10)
                
                for factor, score in report['summary']['quality_factors']:
                    ttk.Label(factors_frame, 
                             text=f"{factor}: {score:.1f}",
                             font=("Arial", 11)).pack(anchor=tk.W, pady=2)
            
            # Add warnings
            if report['warnings']:
                warnings_frame = ttk.LabelFrame(summary_frame, text="Warnings", padding=10)
                warnings_frame.pack(fill=tk.X, pady=10)
                
                for i, warning in enumerate(report['warnings']):
                    ttk.Label(warnings_frame, 
                             text=f"• {warning}",
                             foreground="red",
                             font=("Arial", 11)).pack(anchor=tk.W, pady=2)
            else:
                ttk.Label(summary_frame, 
                         text="No warnings detected",
                         foreground="green",
                         font=("Arial", 11)).pack(anchor=tk.W, pady=10)
            
            # Statistics tab
            stats_frame = ttk.Frame(notebook, padding=10)
            notebook.add(stats_frame, text="Statistics")
            
            # Add T1 statistics
            t1_stats_frame = ttk.LabelFrame(stats_frame, text="T1 Statistics (ms)", padding=10)
            t1_stats_frame.pack(fill=tk.X, pady=10)
            
            if 'statistics' in report:
                for i, (key, value) in enumerate(report['statistics'].items()):
                    label_text = key.replace('t1_', 'T1 ').replace('_', ' ').title()
                    ttk.Label(t1_stats_frame, 
                             text=f"{label_text}: {value:.1f}" if isinstance(value, float) else f"{label_text}: {value}",
                             font=("Arial", 11)).pack(anchor=tk.W, pady=2)
            
            # Add quality metrics
            quality_stats_frame = ttk.LabelFrame(stats_frame, text="Quality Metrics", padding=10)
            quality_stats_frame.pack(fill=tk.X, pady=10)
            
            if 'quality' in report:
                for i, (key, value) in enumerate(report['quality'].items()):
                    if key == 'model_types':
                        # Special handling for model types dictionary
                        ttk.Label(quality_stats_frame, 
                                 text=f"Model Types:",
                                 font=("Arial", 11)).pack(anchor=tk.W, pady=2)
                        
                        for model, count in value.items():
                            ttk.Label(quality_stats_frame, 
                                     text=f"  • {model}: {count} pixels",
                                     font=("Arial", 10)).pack(anchor=tk.W, padx=20)
                    else:
                        label_text = key.replace('_', ' ').title()
                        ttk.Label(quality_stats_frame, 
                                 text=f"{label_text}: {value:.2f}" if isinstance(value, float) else f"{label_text}: {value}",
                                 font=("Arial", 11)).pack(anchor=tk.W, pady=2)
            
            # Sequence tab
            seq_frame = ttk.Frame(notebook, padding=10)
            notebook.add(seq_frame, text="Sequence")
            
            # Add sequence parameters
            seq_params_frame = ttk.LabelFrame(seq_frame, text="Sequence Parameters", padding=10)
            seq_params_frame.pack(fill=tk.X, pady=10)
            
            if 'sequence' in report:
                for i, (key, value) in enumerate(report['sequence'].items()):
                    if value is not None:
                        label_text = key.replace('_', ' ').title()
                        ttk.Label(seq_params_frame, 
                                 text=f"{label_text}: {value}",
                                 font=("Arial", 11)).pack(anchor=tk.W, pady=2)
            
            # Add buttons
            button_frame = ttk.Frame(report_window, padding=10)
            button_frame.pack(fill=tk.X)
            
            ttk.Button(button_frame, text="Save Report", 
                      command=lambda: self.save_quality_report(report)).pack(side=tk.RIGHT, padx=5)
            
            ttk.Button(button_frame, text="Close", 
                      command=report_window.destroy).pack(side=tk.RIGHT, padx=5)
            
            self.update_status("Quality report generated")
            
        except Exception as e:
            self.update_status("Error generating quality report")
            show_error_dialog("Quality Report Error", "Error generating quality report", e)

    def save_quality_report(self, report):
        """Save quality report to file"""
        try:
            # Ask for the file path
            file_path = filedialog.asksaveasfilename(
                title="Save Quality Report",
                defaultextension=".json",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return
            
            # Add timestamp
            report['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save the report
            import json
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            messagebox.showinfo("Success", f"Quality report saved to {file_path}")
            
        except Exception as e:
            show_error_dialog("Save Error", "Error saving quality report", e)

    def save_t1_data(self):
        """Save T1 map data to CSV/Excel file with metadata"""
        if self.t1_map is None or self.roi_coords is None:
            messagebox.showinfo("Info", "Calculate T1 values first")
            return
        
        try:
            # Ask for the file path
            file_path = filedialog.asksaveasfilename(
                title="Save T1 Data",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel files", "*.xlsx"),
                    ("CSV files", "*.csv"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                self.update_status("Save cancelled")
                return
            
            # Get file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            # Update status
            self.update_status(f"Saving T1 data to {os.path.basename(file_path)}...")
            
            # Create metadata
            metadata = {
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ROI_Coordinates": f"x1={self.roi_coords[0]}, x2={self.roi_coords[1]}, y1={self.roi_coords[2]}, y2={self.roi_coords[3]}",
                "ROI_Size": f"{self.roi_coords[1] - self.roi_coords[0]}×{self.roi_coords[3] - self.roi_coords[2]} pixels",
                "Mean_ROI_T1": f"{self.t1_value:.2f} ms",
                "TI_Values_ms": ", ".join([f"{ti:.1f}" for ti in self.tis]),
                "TR_Value_ms": str(self.sequence_params.get('TR', 'Unknown')),
                "Flip_Angle_degrees": str(self.sequence_params.get('FlipAngle', 'Unknown')),
                "Inversion_Efficiency": str(calculate_inversion_efficiency(self.sequence_params.get('FlipAngle', 180)))
            }
            
            # Calculate statistics for valid T1s
            valid_t1s = self.t1_map[~np.isnan(self.t1_map)]
            if len(valid_t1s) > 0:
                metadata["T1_Mean_ms"] = f"{np.mean(valid_t1s):.2f}"
                metadata["T1_Median_ms"] = f"{np.median(valid_t1s):.2f}"
                metadata["T1_StdDev_ms"] = f"{np.std(valid_t1s):.2f}"
                metadata["T1_Min_ms"] = f"{np.min(valid_t1s):.2f}"
                metadata["T1_Max_ms"] = f"{np.max(valid_t1s):.2f}"
                metadata["T1_Successful_Fits_Percentage"] = f"{len(valid_t1s) / self.t1_map.size * 100:.2f}%"
            
            # Save based on file extension
            if ext == '.xlsx':
                try:
                    import pandas as pd
                    
                    # Create a Pandas Excel writer
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        # Write metadata to a sheet
                        pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value']).to_excel(
                            writer, sheet_name='Metadata', index=False
                        )
                        
                        # Convert T1 map to dataframe with coordinates
                        t1_data = []
                        for y in range(self.t1_map.shape[0]):
                            for x in range(self.t1_map.shape[1]):
                                t1_value = self.t1_map[y, x]
                                if not np.isnan(t1_value):
                                    # Calculate actual image coordinates
                                    img_x = x + self.roi_coords[0]
                                    img_y = y + self.roi_coords[2]
                                    t1_data.append({
                                        'ROI_X': x,
                                        'ROI_Y': y,
                                        'Image_X': img_x,
                                        'Image_Y': img_y,
                                        'T1_ms': t1_value  # Save raw T1 in ms without scaling
                                    })
                        
                        # Write T1 data to sheet
                        pd.DataFrame(t1_data).to_excel(
                            writer, sheet_name='T1_Values', index=False
                        )
                        
                        # Export the full T1 map as a matrix
                        pd.DataFrame(self.t1_map).to_excel(
                            writer, sheet_name='T1_Map', index=True
                        )
                        
                        # Save fit data for mean ROI
                        mean_fit_data = pd.DataFrame({
                            'TI_ms': self.tis,
                            'Signal': self.processed_signal,
                            'Fitted': self.fitted_curve
                        })
                        mean_fit_data.to_excel(writer, sheet_name='Mean_ROI_Fit', index=False)
                    
                    self.update_status(f"T1 data saved to {os.path.basename(file_path)}")
                    messagebox.showinfo("Success", f"T1 data successfully saved to {file_path}")
                    
                except ImportError:
                    messagebox.showwarning("Missing Library", 
                                         "Pandas or openpyxl is not installed. Saving as CSV instead.")
                    # Fall back to CSV if pandas/openpyxl not available
                    ext = '.csv'
            
            if ext == '.csv' or ext == '.txt':
                # Open the file for writing
                with open(file_path, 'w', newline='') as f:
                    # Write metadata
                    f.write("# T1 Mapping Results\n")
                    f.write("# ==================\n")
                    for key, value in metadata.items():
                        f.write(f"# {key}: {value}\n")
                    
                    f.write("\n# T1 Map Matrix (ms) - row, column format\n")
                    
                    # Write the T1 map as a CSV matrix
                    for row in self.t1_map:
                        f.write(",".join([f"{val:.2f}" if not np.isnan(val) else "NaN" for val in row]) + "\n")
                    
                    f.write("\n# Pixel-by-pixel T1 Values (ms)\n")
                    f.write("ROI_X,ROI_Y,Image_X,Image_Y,T1_ms\n")
                    
                    # Write each pixel's data
                    for y in range(self.t1_map.shape[0]):
                        for x in range(self.t1_map.shape[1]):
                            t1_value = self.t1_map[y, x]
                            if not np.isnan(t1_value):
                                # Calculate actual image coordinates
                                img_x = x + self.roi_coords[0]
                                img_y = y + self.roi_coords[2]
                                f.write(f"{x},{y},{img_x},{img_y},{t1_value:.2f}\n")
                    
                    f.write("\n# Mean ROI Fit Data\n")
                    f.write("TI_ms,Signal,Fitted\n")
                    
                    # Write the mean ROI fit data
                    for i in range(len(self.tis)):
                        f.write(f"{self.tis[i]:.2f},{self.processed_signal[i]:.6f},{self.fitted_curve[i]:.6f}\n")
                
                self.update_status(f"T1 data saved to {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"T1 data successfully saved to {file_path}")
        
        except Exception as e:
            self.update_status("Error saving T1 data")
            show_error_dialog("Save Error", "Error saving T1 data", e)
            print(f"Save error: {e}")

    # Final method to enable the save data button after adding save_t1_data
    def enable_save_data_button(self):
        """Enable the save data button once save_t1_data method is available"""
        if hasattr(self, 'save_data_button'):
            self.save_data_button.configure(state=tk.NORMAL)

    # Add this to the __init__ method after creating UI elements:
    # self.enable_save_data_button()

def main():
    try:
        root = tk.Tk()
        root.title("Enhanced GRE MOLLI T1 Mapping Tool")
        root.geometry("1500x700")  # Make window wider to accommodate the new panel
        app = LeanT1MappingApp(root)
        
        # Enable save data button now that save_t1_data method exists
        if hasattr(app, 'save_data_button'):
            app.save_data_button.configure(state=tk.DISABLED)  # Will be enabled after T1 calculation
        
        def on_closing():
            print("Shutting down...")
            try:
                root.quit()
                root.destroy()
            except:
                pass
            import os
            os._exit(0)
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        
    except Exception as e:
        error_msg = f"Critical error during startup: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        try:
            import tkinter.messagebox as msgbox
            msgbox.showerror("Startup Error", 
                           f"Failed to start T1 Mapping Tool:\n\n{error_msg}")
        except:
            pass

if __name__ == "__main__":
    main()
