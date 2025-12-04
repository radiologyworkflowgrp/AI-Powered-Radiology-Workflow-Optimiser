import { Toaster } from "@/components/ui/toaster.jsx";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { UserProvider } from "./context/UserContext";
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";
import { useEffect, useState } from "react";
import { useUser } from "./hooks/use-user";

import Login from "./pages/Login";
import Signup from "./pages/signup";

// Admin Pages
import AdminDashboard from "./pages/admin/AdminDashboard";
import DoctorList from "./pages/admin/DoctorList";
import Log from "./pages/admin/Log";
import PatientStatus from "./pages/admin/PatientStatus";
import PatientAddPortal from "./pages/admin/PatientAddPortal";
import DoctorAddPortal from "./pages/admin/DoctorAddPortal";


//Doctor Pages
import DoctorDashboard from "./pages/doctor/DoctorDashboard";
import DoctorNotes from "./pages/doctor/DoctorNotes";

//Patient Pages
import PatientDashboard from "./pages/patient/PatientDashboard";
import PatientNotes from "./pages/patient/PatientNotes";
import PatientInfoPortal from "./pages/patient/patientinfoportal";

//Common Pages
import RadiologyResults from "./pages/RadiologyResults";
import Precriptions from "./pages/Precriptions";
import DicomUploadPage from "./pages/DicomUploadPage";

// Not Found Page
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

// Auth checker component
const AuthChecker = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { setRole, setUser } = useUser();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      // Skip auth check if already on login or signup page
      if (location.pathname === "/" || location.pathname === "/signup") {
        setIsChecking(false);
        return;
      }

      try {
        const response = await fetch("http://localhost:3002/api/auth/verify", {
          method: "GET",
          credentials: "include", // Important: send cookies
        });

        if (response.ok) {
          const data = await response.json();
          // User is authenticated
          setRole(data.user.role);
          setUser(data.user);
          console.log("Auto-login successful:", data.user);
        } else {
          // Not authenticated, redirect to login if not already there
          if (location.pathname !== "/" && location.pathname !== "/signup") {
            navigate("/");
          }
        }
      } catch (error) {
        console.error("Auth check error:", error);
        // On error, redirect to login if not already there
        if (location.pathname !== "/" && location.pathname !== "/signup") {
          navigate("/");
        }
      } finally {
        setIsChecking(false);
      }
    };

    checkAuth();
  }, [location.pathname, navigate, setRole, setUser]);

  // Show loading while checking auth
  if (isChecking && location.pathname !== "/" && location.pathname !== "/signup") {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Checking authentication...</p>
        </div>
      </div>
    );
  }

  return children;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <UserProvider>
        <BrowserRouter>
          <ErrorBoundary>
            <AuthChecker>
              <Routes>
                <Route path="/" element={<Login />} />
                {/* Sign up*/}
                <Route path="/signup" element={<Signup />} />

                {/* Admin Routes */}
                <Route path="/admin/dashboard" element={<AdminDashboard />} />
                <Route path="/admin/doctors" element={<DoctorList />} />
                <Route path="/admin/patient-status" element={<PatientStatus />} />
                <Route path="/admin/add-patient" element={<PatientAddPortal />} />
                <Route path="/admin/add-doctor" element={<DoctorAddPortal />} />
                <Route path="/admin/logs" element={<Log />} />
                {/* Doctor Routes */}
                <Route path="/doctor/dashboard" element={<DoctorDashboard />} />
                <Route path="/doctor/notes" element={<DoctorNotes />} />
                {/* Patient Routes */}
                <Route path="/patient/dashboard" element={<PatientDashboard />} />
                <Route path="/patient/notes" element={<PatientNotes />} />
                <Route path="/add-patient" element={<PatientInfoPortal />} />
                {/* Common Routes */}
                <Route path="/radiology-results" element={<RadiologyResults />} />
                <Route path="/prescriptions" element={<Precriptions />} />
                <Route path="/dicom-upload" element={<DicomUploadPage />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </AuthChecker>
          </ErrorBoundary>
        </BrowserRouter>
      </UserProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
