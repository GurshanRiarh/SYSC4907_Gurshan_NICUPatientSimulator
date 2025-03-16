// static/js/main.js

// =============================================================================
// NICU Patient Simulator - Main JavaScript File
// This file contains helper functions to dynamically populate form fields 
// (Age and Condition dropdowns) based on the simulation mode, and sets up
// event listeners for form submission and help modal display.
// =============================================================================

// -----------------------------------------------------------------------------
 // Helper Function: getSimulationMode
 // Reads the "mode" parameter from the URL and returns its value.
 // Defaults to 'preterm' if not specified.
 // -----------------------------------------------------------------------------
 function getSimulationMode() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get('mode') || 'preterm';
}

// -----------------------------------------------------------------------------
// Function: updateAgeDropdown
// Populates the Age dropdown based on the current simulation mode.
// - For preterm: creates options in the format "week/day" (22/1 to 40/6).
// - For neonate: creates options for age in days (1 to 28).
// -----------------------------------------------------------------------------
function updateAgeDropdown() {
  const simulationMode = getSimulationMode();
  const ageDropdown = document.getElementById("Age");
  ageDropdown.innerHTML = ""; // Clear existing options
  
  if (simulationMode === "preterm") {
      // Generate options for gestational age (weeks/days)
      for (let week = 22; week <= 40; week++) {
          for (let day = 1; day <= 6; day++) {
              const option = document.createElement("option");
              option.value = `${week}/${day}`;
              option.textContent = `${week} weeks / ${day} days`;
              ageDropdown.appendChild(option);
          }
      }
  } else {
      // Generate options for age in days
      for (let day = 1; day <= 28; day++) {
          const option = document.createElement("option");
          option.value = day;
          option.textContent = `${day} day${day > 1 ? 's' : ''}`;
          ageDropdown.appendChild(option);
      }
  }
}

// -----------------------------------------------------------------------------
// Function: updateConditionDropdown
// Dynamically populates the Condition dropdown based on the simulation mode.
// - For preterm: Only healthy condition is allowed, so the dropdown is disabled.
// - For neonate: Provides multiple condition options.
// -----------------------------------------------------------------------------
function updateConditionDropdown() {
  const simulationMode = getSimulationMode();
  const conditionDropdown = document.getElementById("condition");
  conditionDropdown.innerHTML = ""; // Clear existing options

  if (simulationMode === "preterm") {
      // For preterm simulation: force healthy condition only.
      const option = document.createElement("option");
      option.value = "none";
      option.textContent = "None";
      conditionDropdown.appendChild(option);
      conditionDropdown.disabled = true;
  } else {
      // For neonate simulation: offer multiple options.
      conditionDropdown.disabled = false;
      const options = [
          { value: "normal", text: "None" },
          { value: "bradycardia", text: "Bradycardia" },
          { value: "tachycardia", text: "Tachycardia" },
          { value: "both", text: "Bradycardia & Tachycardia" }
      ];
      options.forEach(function(opt) {
          const option = document.createElement("option");
          option.value = opt.value;
          option.textContent = opt.text;
          conditionDropdown.appendChild(option);
      });
  }
}

// -----------------------------------------------------------------------------
// DOMContentLoaded Event Listener
// Sets up initial form values, populates dropdowns, and adds event listeners
// for form submission and help modal display.
// -----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function() {
  // Set hidden input with the current simulation mode for backend use.
  const simulationMode = getSimulationMode();
  document.getElementById('simulation_mode').value = simulationMode;
  
  // Populate the Age and Condition dropdowns.
  updateAgeDropdown();
  updateConditionDropdown();
  
  // Setup form submission: display loading spinner and disable submit button.
  const form = document.getElementById('simulationForm');
  const submitBtn = document.getElementById('submitBtn');
  if (form) {
    form.addEventListener('submit', function(event) {
      submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running Simulation...';
      submitBtn.disabled = true;
    });
  }
  
  // Setup help button: display the help modal when clicked.
  const helpBtn = document.getElementById('helpBtn');
  if (helpBtn) {
    helpBtn.addEventListener('click', function() {
      const helpModal = new bootstrap.Modal(document.getElementById('helpModal'));
      helpModal.show();
    });
  }
});
