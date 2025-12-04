import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

const Log = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Redirect to the new Logs page
    navigate("/admin/logs");
  }, [navigate]);

  return null;
};

export default Log;