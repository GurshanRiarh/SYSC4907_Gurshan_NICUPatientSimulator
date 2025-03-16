// static/js/result.js

// =============================================================================
// NICU Patient Simulator - Results Page Script
// This script initializes the results page functionality, including loading 
// vital signs plot images, extracting the patient ID from the URL, handling 
// tab switching, and initializing Bootstrap components.
// =============================================================================
document.addEventListener('DOMContentLoaded', function() {

    // -----------------------------------------------------------------------------
    // Function: loadVitalSignsPlot
    // Purpose: Loads the static vital signs plot image for the patient.
    // It extracts the patient ID from the URL and sets the 'src' attribute 
    // of the image element with ID 'vitalSignsImage'. If the image fails to load,
    // a placeholder image is displayed instead.
    // -----------------------------------------------------------------------------
    function loadVitalSignsPlot() {
        const patientId = getPatientIdFromUrl();
        const img = document.getElementById('vitalSignsImage');
        
        if (img) {
            // Set image source to the expected static plot location
            img.src = `/static/plots/${patientId}/static_vital_signs.png`;
            
            // If image fails to load, use a placeholder image and update the alt text.
            img.onerror = function() {
                img.src = "/static/img/plot-placeholder.png";
                img.alt = "Plot not available";
            };
        }
    }
    
    // -----------------------------------------------------------------------------
    // Helper Function: getPatientIdFromUrl
    // Purpose: Extracts the patient ID from the current URL path.
    // The function searches the URL segments for the "result" keyword and returns
    // the subsequent segment as the patient ID.
    // -----------------------------------------------------------------------------
    function getPatientIdFromUrl() {
        const pathParts = window.location.pathname.split('/');
        for (let i = 0; i < pathParts.length; i++) {
            if (pathParts[i] === 'result' && i + 1 < pathParts.length) {
                return pathParts[i + 1];
            }
        }
        return '';
    }
    
    // -----------------------------------------------------------------------------
    // Tab Event Handling
    // -----------------------------------------------------------------------------
    // When the "plots" tab is shown, load the vital signs plot image.
    const plotsTab = document.getElementById('plots-tab');
    if (plotsTab) {
        plotsTab.addEventListener('shown.bs.tab', function(event) {
            loadVitalSignsPlot();
        });
    }
    
    // Initialize all tabs using Bootstrap's API.
    // For each tab link in the '#resultTabs' container, add an event listener 
    // to prevent the default behavior and show the corresponding tab.
    const triggerTabList = [].slice.call(document.querySelectorAll('#resultTabs a'));
    triggerTabList.forEach(function(triggerEl) {
        const tabTrigger = new bootstrap.Tab(triggerEl);
        triggerEl.addEventListener('click', function(event) {
            event.preventDefault();
            tabTrigger.show();
        });
    });
});
