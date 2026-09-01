export default {
  // Brand
  brandName: "flockcare",
  appTitle: "FlockCare",
  appBadge: "Poultry Bioacoustic AI",
  subtitle: "Precision respiratory screening for poultry health",

  // Landing / Onboarding (Hers inspired)
  landingHeroPrefix: "Better care designed just for ",
  landingHeroHighlight: "your flock",
  landingSubtitle: "Personalized bioacoustic respiratory screening and disease differential diagnostics.",
  getStarted: "Get started",
  logIn: "Log in",
  newToApp: "New to FlockCare?",
  createAccount: "Create an account",
  trustedBadge: "Trusted by poultry producers across 50,000+ birds",

  // Feature cards on landing
  featureRespiratoryTitle: "Respiratory Health",
  featureRespiratoryDesc: "Early rale & wheeze detection",
  featureDifferentialTitle: "Disease Differential",
  featureDifferentialDesc: "IBV, CRD & Coryza matching",
  featureVetReportsTitle: "Veterinary Reports",
  featureVetReportsDesc: "Clinical export for veterinarians",

  // Sign In
  signInTitle: "Welcome back",
  signInSubtitle: "Sign in to access flock surveillance records and veterinary reports.",
  emailOrFarmLabel: "Farm identifier or Email",
  emailPlaceholder: "e.g. greenvalley_coop or farm@example.com",
  passwordLabel: "Password / Passcode",
  passwordPlaceholder: "••••••••",
  rememberDevice: "Remember this farm device",
  forgotPassword: "Forgot password?",
  signInBtn: "Sign in",
  continueAsGuest: "Continue as Guest Farmer",
  noAccountPrompt: "Don't have a farm account?",
  signUpLink: "Sign up",

  // Dashboard & Record
  dashboardGreeting: "Welcome back,",
  defaultFarmName: "Valley Crest Coop",
  activeSurveillance: "Coop Acoustic Surveillance",
  recordHeading: "Coop Health Screening",
  recordInstructions: "Position your device 1–2 meters from the flock. Capture 15–30 seconds of ambient vocalizations.",
  minDurationNote: "Minimum 5 seconds required. 15–30s recommended for maximum multi-window accuracy.",
  tapToRecord: "Start Coop Screening",
  tapToStop: "Finish & Analyze Recording",
  recordingActive: "Listening to flock bioacoustics…",
  uploadFile: "Or upload an audio recording",

  // Analyzing
  analyzingHeading: "Analyzing Coop Bioacoustics",
  analyzingSubtitle: "Neural model scanning 5-second spectrogram windows for tracheal wheezes and rales…",
  analyzingStep1: "Decoding audio container…",
  analyzingStep2: "Extracting log-mel spectrogram features…",
  analyzingStep3: "Evaluating acoustic domain & OOD gating…",
  analyzingStep4: "Synthesizing multi-window clinical risk…",

  // Results
  healthyTitle: "Flock is Healthy",
  stressTitle: "Signs of Respiratory Stress Detected",
  elevatedTitle: "Elevated Respiratory Distress Risk",
  healthyDesc: "Acoustic profile indicates normal, unlabored flock vocalizations.",
  stressDesc: "Subtle wheezing or vocal strain detected. Monitor the flock closely over the next 24–48 hours.",
  elevatedDesc: "Elevated respiratory distress patterns detected. Isolate affected birds and consult a poultry veterinarian.",
  outOfRangeTitle: "Acoustic Range Notice",
  outOfRangeWarning: "Audio characteristics deviate from calibrated baseline. Recommendation: Re-record closer to the birds in a quiet environment.",
  findVet: "Find Nearby Poultry Veterinarians",
  checkAgain: "Record Another Screening",
  windowsAnalyzed: "Acoustic Windows Analyzed",
  riskScore: "Risk Level",

  // History & Records
  historyTitle: "Screening History",
  historyEmpty: "No previous screenings recorded on this device.",
  historyDeviceOnly: "History is stored securely on your device.",
  clearHistory: "Clear History",

  // Disclaimer & Errors
  disclaimer: "FlockCare is a bioacoustic AI screening tool designed to augment poultry management. It does not replace formal clinical diagnosis by a licensed veterinarian.",
  offlineWarning: "Connecting to AI backend (waking up cloud server, please wait a moment...)",
  micErrorDenied: "Microphone access was denied. Please enable microphone permissions in your browser settings.",
  micErrorNotFound: "No microphone detected on this device.",
  micErrorBusy: "The microphone is in use by another application.",
};
