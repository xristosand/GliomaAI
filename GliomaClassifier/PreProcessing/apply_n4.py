
import time
import SimpleITK as sitk

# =============================================================================
# 1. (N4 BIAS CORRECTION)
# =============================================================================
def apply_n4_algorithm(sitk_image, choice=1):

    print("N4 Function called")
    
    """
    Εφαρμόζει N4 Bias Field Correction με δύο τρόπους (Modes).
    Διατηρεί ΑΥΣΤΗΡΑ τα Metadata (Spacing, Origin, Direction).
    
    Args:
        choice (int): 
            0 -> Native SimpleITK (Πιο αργό, τρέχει σε Full Resolution αν δεν υποστηρίζεται το shrink).
            1 -> Manual Cascade (Γρήγορο & Βέλτιστο). Κάνει Shrink 2 -> N4 -> Full Res -> N4.
    """

    # Μετατροπή σε Float32 (απαραίτητο για N4)
    image_float = sitk.Cast(sitk_image, sitk.sitkFloat32)

    # ---------------------------------------------------------------------
    # MODE 0: NATIVE SIMPLEITK (Standard)
    # ---------------------------------------------------------------------
    if choice == 0:      
        start_time = time.time()

        # Μάσκα Otsu
        mask = sitk.OtsuThreshold(image_float, 0, 1, 200)
        
        # Ρύθμιση N4
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 30, 20])
        
        # Προσπάθεια χρήσης built-in Shrinking
        try:
            corrector.SetShrinkFactorsPerLevel([2, 1, 1])
        except AttributeError:
            pass # Αν δεν υπάρχει η εντολή, απλά τρέχει σε Full Res (πιο αργό)

        # Execution
        corrected_image = corrector.Execute(image_float, mask)
        
        elapsed = time.time() - start_time
        print(f"     Execution time: {elapsed:.2f} sec")

        return corrected_image
    
    # ---------------------------------------------------------------------
    # MODE 1: MANUAL CASCADE (Fast & Safe - Recommended)
    # ---------------------------------------------------------------------
    elif choice == 1:
        start_time = time.time()
        
        # --- STAGE 1: COARSE CORRECTION (Scale 1/2) ---
        # 1. Manual Shrink (faster)
        shrink_filter = sitk.ShrinkImageFilter()
        shrink_filter.SetShrinkFactor(2)
        image_small = shrink_filter.Execute(image_float)
        
        # 2. Mask & N4 (Small)
        mask_small = sitk.OtsuThreshold(image_small, 0, 1, 200)
        
        corrector_stage1 = sitk.N4BiasFieldCorrectionImageFilter()
        corrector_stage1.SetMaximumNumberOfIterations([50, 50, 50]) 
        
        # Run shrinked image
        corrector_stage1.Execute(image_small, mask_small)
        
        # 3. Upsample Bias Field & Apply
        log_bias_1 = corrector_stage1.GetLogBiasFieldAsImage(image_small)
        # Upsample με BSpline χρησιμοποιώντας την original ως reference
        log_bias_1_full = sitk.Resample(log_bias_1, image_float, sitk.Transform(), sitk.sitkBSpline)
        bias_field_1 = sitk.Exp(log_bias_1_full)
        
        # Προσωρινό αποτέλεσμα (Stage 1 Corrected)
        image_stage1 = image_float / bias_field_1
        
        # --- STAGE 2: FINE TUNING (Full Scale) ---
        # Τώρα δουλεύουμε στην image_stage1 που είναι ήδη "καθαρή" χοντρικά
        
        # 1. Mask (Otsu on Full Res)
        mask_full = sitk.OtsuThreshold(image_stage1, 0, 1, 200)
        
        # 2. N4 (50 Iters for details)
        corrector_stage2 = sitk.N4BiasFieldCorrectionImageFilter()
        corrector_stage2.SetMaximumNumberOfIterations([50]) 
        
        # Εκτέλεση (Αυτό παίρνει τον περισσότερο χρόνο)
        image_final = corrector_stage2.Execute(image_stage1, mask_full)
        
        elapsed = time.time() - start_time
        print(f"     Execution time: {elapsed:.2f} sec")

        return image_final
    
    else:
        raise ValueError("choice must be 0 or 1")