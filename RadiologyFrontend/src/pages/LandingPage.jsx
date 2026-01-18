import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 relative overflow-hidden">
      {/* Background Gradients */}
      <div className="absolute top-0 -left-4 w-96 h-96 bg-primary/20 rounded-full blur-3xl opacity-50 pointer-events-none" />
      <div className="absolute bottom-0 -right-4 w-96 h-96 bg-accent/20 rounded-full blur-3xl opacity-50 pointer-events-none" />

      {/* Navigation Bar */}
      <nav className="flex items-center justify-between px-8 py-6 w-full max-w-7xl mx-auto backdrop-blur-sm relative z-10">
        <div className="flex items-center gap-2">
          {/* Logo Icon Placeholder */}
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="w-5 h-5 text-primary-foreground"
            >
              <path d="M12 2a10 10 0 1 0 10 10H12V2z" />
              <path d="M12 12 2.1 12.1" />
              <path d="M12 12l8.9 8.9" />
            </svg>
          </div>
          <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-primary">
            RadiologyFlow
          </span>
        </div>

        <Button
          onClick={() => navigate("/login")}
          className="font-semibold"
          variant="secondary"
        >
          Sign In
        </Button>
      </nav>

      {/* Hero Section */}
      <main className="flex flex-col items-center justify-center min-h-[calc(100vh-88px)] px-4 text-center relative z-10">
        <div className="max-w-4xl space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-1000">

          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight">
            Transforming Healthcare Through <br />
            <span className="text-primary bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-primary">
              Intelligent Imaging
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Streamline your radiology workflow with cutting-edge AI technology.
            Faster diagnoses, better patient outcomes, and seamless collaboration.
          </p>

          <div className="grid md:grid-cols-3 gap-6 text-left mt-16 max-w-5xl mx-auto">
            <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm hover:border-slate-700 transition-colors">
              <div className="w-12 h-12 bg-blue-500/10 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <activity className="w-6 h-6" />
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-slate-100">AI Analysis</h3>
              <p className="text-slate-400">Advanced algorithms to detect anomalies and prioritize urgent cases automatically.</p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm hover:border-slate-700 transition-colors">
              <div className="w-12 h-12 bg-green-500/10 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-green-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                  <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
                  <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-slate-100">Team Collaboration</h3>
              <p className="text-slate-400">Seamless communication between radiologists, doctors, and specialists.</p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm hover:border-slate-700 transition-colors">
              <div className="w-12 h-12 bg-purple-500/10 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-purple-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect width="18" height="18" x="3" y="4" rx="2" ry="2" />
                  <line x1="16" x2="16" y1="2" y2="6" />
                  <line x1="8" x2="8" y1="2" y2="6" />
                  <line x1="3" x2="21" y1="10" y2="10" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-slate-100">Smart Scheduling</h3>
              <p className="text-slate-400">Optimized patient scheduling and resource management for efficiency.</p>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
};

export default LandingPage;

