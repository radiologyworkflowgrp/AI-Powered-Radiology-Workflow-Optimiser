const API_BASE_URL = 'http://localhost:3002/api'; // Example URL for a local backend

/**
 * A helper function to handle responses from the fetch API.
 * It checks if the response was successful and parses the JSON body.
 * @param {Response} response - The response object from a fetch call.
 * @returns {Promise<any>} The JSON data from the response.
 * @throws {Error} Throws an error if the network response was not ok.
 */
const handleResponse = async (response) => {
  if (!response.ok) {
    // Try to parse a JSON error message from the backend
    const errorData = await response.json().catch(() => ({ message: 'An unknown error occurred.' }));
    const errorMessage = errorData.error ? `${errorData.message}: ${errorData.error}` : errorData.message;
    throw new Error(errorMessage || `HTTP error! Status: ${response.status}`);
  }
  return response.json();
};

/**
 * Fetches all radiology results from the backend.
 * @returns {Promise<Array>} A promise that resolves to an array of radiology results.
 */
export const getRadiologyResults = async () => {
  const response = await fetch(`${API_BASE_URL}/radiology-results`, {
    credentials: 'include', // Send cookies (authToken) to backend
  });
  const data = await handleResponse(response);

  // Handle the new response format
  if (data.results && Array.isArray(data.results)) {
    return data.results;
  }

  // Fallback for backward compatibility
  return Array.isArray(data) ? data : [];
};

/**
 * Fetches recent radiology results for a specific patient.
 * @param {string} patientId - The patient ID
 * @param {number} limit - Maximum number of recent reports to fetch
 * @returns {Promise<Array>} A promise that resolves to an array of recent radiology results.
 */
export const getPatientRecentReports = async (patientId, limit = 5) => {
  const response = await fetch(`${API_BASE_URL}/radiology-results/patient/${patientId}/recent?limit=${limit}`);
  const data = await handleResponse(response);

  // Handle the new response format
  if (data.results && Array.isArray(data.results)) {
    return data.results;
  }

  // Fallback for backward compatibility
  return Array.isArray(data) ? data : [];
};

export const getDoctors = async () => {
  const response = await fetch(`${API_BASE_URL}/doctors`);
  return handleResponse(response);
};

export const getDoctorById = async (doctorId) => {
  const response = await fetch(`${API_BASE_URL}/doctors/${doctorId}`);
  return handleResponse(response);
};

export const updateDoctor = async (doctorId, updatedData) => {
  const response = await fetch(`${API_BASE_URL}/doctors/${doctorId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(updatedData),
  });
  return handleResponse(response);
};

export const getPatients = async () => {
  const response = await fetch(`${API_BASE_URL}/patients`);
  return handleResponse(response);
};

export const addPatientQuick = async (patientData) => {
  const response = await fetch(`${API_BASE_URL}/patients`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(patientData),
  });
  return handleResponse(response);
};

export const addPatient = async (patientData) => {
  const isFormData = patientData instanceof FormData;
  const headers = isFormData ? {} : { 'Content-Type': 'application/json' };
  const body = isFormData ? patientData : JSON.stringify(patientData);

  const response = await fetch(`${API_BASE_URL}/patients`, {
    method: 'POST',
    headers: headers,
    body: body,
  });
  return handleResponse(response);
};

export const addDoctor = async (doctorData) => {
  const response = await fetch(`${API_BASE_URL}/doctors`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(doctorData),
  });
  return handleResponse(response);
};

/**

 * @param {object} credentials - The user's login credentials.
 * @param {string} credentials.email - The user's email.
 * @param {string} credentials.password - The user's password.
 * @returns {Promise<object>} A promise that resolves to the user session data (e.g., a token).
 */
export const loginUser = async (credentials) => {
  const response = await fetch(`${API_BASE_URL}/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(credentials),
  });
  return handleResponse(response);
};

export const getPatientById = async (patientId) => {
  const response = await fetch(`${API_BASE_URL}/patients/${patientId}`);
  return handleResponse(response);
};

export const updatePatient = async (patientId, updatedData) => {
  const response = await fetch(`${API_BASE_URL}/patients/${patientId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(updatedData),
  });
  return handleResponse(response);
};

export const updatePatientProfile = async (patientId, profileData) => {
  const response = await fetch(`${API_BASE_URL}/patients/profile/${patientId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(profileData),
  });
  return handleResponse(response);
};

// Prescription API methods
export const getPrescriptions = async (patientName = null) => {
  const url = patientName
    ? `${API_BASE_URL}/prescriptions?patientName=${encodeURIComponent(patientName)}`
    : `${API_BASE_URL}/prescriptions`;
  const response = await fetch(url);
  return handleResponse(response);
};

export const getPendingPrescriptions = async () => {
  const response = await fetch(`${API_BASE_URL}/prescriptions/pending`);
  return handleResponse(response);
};

export const createPrescription = async (prescriptionData) => {
  const response = await fetch(`${API_BASE_URL}/prescriptions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(prescriptionData),
  });
  return handleResponse(response);
};

export const requestRefill = async (prescriptionId) => {
  const response = await fetch(`${API_BASE_URL}/prescriptions/${prescriptionId}/refill`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
  });
  return handleResponse(response);
};

export const approvePrescription = async (prescriptionId) => {
  const response = await fetch(`${API_BASE_URL}/prescriptions/${prescriptionId}/approve`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
  });
  return handleResponse(response);
};

export const rejectPrescription = async (prescriptionId) => {
  const response = await fetch(`${API_BASE_URL}/prescriptions/${prescriptionId}/reject`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
  });
  return handleResponse(response);
};

// Note API methods
export const getNotes = async () => {
  const response = await fetch(`${API_BASE_URL}/notes`);
  return handleResponse(response);
};

export const createNote = async (noteData) => {
  const response = await fetch(`${API_BASE_URL}/notes`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(noteData),
  });
  return handleResponse(response);
};

// Logs API methods
export const getLogs = async (limit = 50, type = 'all') => {
  const response = await fetch(`${API_BASE_URL}/logs?limit=${limit}&type=${type}`);
  return handleResponse(response);
};
