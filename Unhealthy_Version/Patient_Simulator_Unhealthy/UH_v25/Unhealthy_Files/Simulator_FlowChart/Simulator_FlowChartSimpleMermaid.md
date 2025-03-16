graph TD
    A[Start] -->|User Accesses Web UI| B[User Interface]

    B --> C{Mode Selection}
    C -->|Preterm (Healthy)| D[Preterm Simulation]
    C -->|Neonate (Unhealthy)| E[Neonate Simulation]

    B --> F[Patient Information Input]
    F --> G[Enter Patient ID, Start Time, Age, Weight, Gender]
    F --> H[Select Condition (Normal, Bradycardia, Tachycardia, Both)]
    F --> I[Choose Interventions]

    D --> J[Instantiate Healthy Simulator]
    E --> K[Instantiate Unhealthy Simulator]

    J --> L[Load GAN Models]
    K --> L
    L --> M[Generate Synthetic Data]
    M --> N[Load Pre-trained GAN Models]
    N --> O[Heart Rate & Respiration Generators]
    K --> P[Bradycardia/Tachycardia Generators]

    M --> Q[Generate Vital Signs]
    Q --> R[Apply Noise & Normalization]
    R --> S[Smoothing & Delta Calculation]

    S --> T[Schedule Interventions & Assess Pain]
    T --> U[Scheduled (Feeding, Medication, etc.)]
    T --> V[Random Interventions (Poisson Distribution)]
    T --> W[Determine Pain Level]

    W --> X[Export Data]
    X --> Y[Save JSON File (Patient & Simulation Data)]
    X --> Z[Save CSV Files (GAN Outputs & Interventions)]

    X --> AA[Generate Visualization]
    AA --> AB[Static Plots (Matplotlib PNGs)]
    AA --> AC[Interactive Plots (Plotly HTML)]

    AC --> AD[Display Web Application (Flask)]
    AD --> AE[Show Simulation Results]
    AD --> AF[Provide Download Options for Data & Plots]

    AF --> AG[End]
