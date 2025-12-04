import { useUser } from "@/hooks/use-user";
import { Button } from "@/components/ui/button";
import { useLocation, useNavigate } from "react-router-dom";
import { LogOut, FileText, Activity, FileStack, Home } from "lucide-react";
import { useEffect } from "react";

const DashboardLayout = ({ children, title }) => {
  const navigate = useNavigate();
  const { role, setRole, user, setUser } = useUser();
  const location = useLocation();

  useEffect(() => {
    const currentPath = location.pathname;
    if (!role) {
      if (
        currentPath.startsWith("/admin") ||
        currentPath.startsWith("/doctor") ||
        currentPath.startsWith("/patient")
      ) {
        navigate("/");
      }
    } else {
      if (currentPath.startsWith("/admin") && role !== "admin") {
        navigate("/");
      }
      if (currentPath.startsWith("/doctor") && role !== "doctor") {
        navigate("/");
      }
      if (currentPath.startsWith("/patient") && role !== "patient") {
        navigate("/");
      }
    }
  }, [role, location, navigate]);

  const handleLogout = async () => {
    try {
      await fetch("http://localhost:3002/api/auth/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch (error) {
      console.error("Logout failed:", error);
    } finally {
      setRole(null);
      setUser(null);
      navigate("/");
    }
  };

  const doctorLinks = [
    { icon: Home, label: "Dashboard", path: "/doctor/dashboard" },
    { icon: FileText, label: "Medical Notes", path: "/doctor/notes" },
    { icon: Activity, label: "Radiology Results", path: "/radiology-results" },
    { icon: FileStack, label: "Prescriptions", path: "/prescriptions" },
  ];

  const patientLinks = [
    { icon: Home, label: "Dashboard", path: "/patient/dashboard" },
    { icon: FileText, label: "My Notes", path: "/patient/notes" },
    { icon: Activity, label: "Radiology Results", path: "/radiology-results" },
    { icon: FileStack, label: "Prescriptions", path: "/prescriptions" },
    { icon: FileStack, label: "Add Patient", path: "/add-patient" },
  ];

  const adminLinks = [
    { icon: Home, label: "Dashboard", path: "/admin/dashboard" },
    { icon: FileText, label: "Doctors", path: "/admin/doctors" },
    {
      icon: Activity,
      label: "Patient Status",
      path: "/admin/patient-status",
    },
    { icon: FileStack, label: "Add Patient", path: "/admin/add-patient" },
    { icon: Activity, label: "Radiology Results", path: "/radiology-results" },
    { icon: LogOut, label: "Logs", path: "/admin/logs" },
  ];

  let links = [];
  if (role === "admin") {
    links = adminLinks;
  } else if (role === "doctor") {
    links = doctorLinks;
  } else if (role === "patient") {
    links = patientLinks;
  }
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-primary">Radiology Portal</h1>
          <p>
            Welcome {role}, {user ? user.name : ""}
          </p>
          <Button variant="ghost" onClick={handleLogout}>
            <LogOut className="mr-2 h-4 w-4" />
            Logout
          </Button>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <div className="flex gap-6">
          <aside className="w-64 shrink-0">
            <nav className="space-y-2">
              {links.map((link) => (
                <Button
                  key={link.path}
                  variant="ghost"
                  className="w-full justify-start"
                  onClick={() => navigate(link.path)}
                >
                  <link.icon className="mr-2 h-4 w-4" />
                  {link.label}
                </Button>
              ))}
            </nav>
          </aside>

          <main className="flex-1">
            <h2 className="text-3xl font-bold mb-6">{title}</h2>
            {children}
          </main>
        </div>
      </div>
    </div>
  );
};

export { DashboardLayout };
