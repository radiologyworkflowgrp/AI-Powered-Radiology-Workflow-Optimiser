import { createContext, useState, useEffect } from "react";

export const UserContext = createContext();

export const UserProvider = ({ children }) => {
  const [role, setRole] = useState(() => {
    try {
      const item = window.sessionStorage.getItem("userRole");
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.warn("Error reading sessionStorage key “userRole”:", error);
      return null;
    }
  });

  const [user, setUser] = useState(() => {
    try {
      const item = window.sessionStorage.getItem("user");
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.warn("Error reading sessionStorage key “user”:", error);
      return null;
    }
  });

  useEffect(() => {
    try {
      if (role) {
        window.sessionStorage.setItem("userRole", JSON.stringify(role));
      } else {
        window.sessionStorage.removeItem("userRole");
      }
    } catch (error) {
      console.warn("Error setting sessionStorage key “userRole”:", error);
    }
  }, [role]);

  useEffect(() => {
    try {
      if (user) {
        window.sessionStorage.setItem("user", JSON.stringify(user));
      } else {
        window.sessionStorage.removeItem("user");
      }
    } catch (error) {
      console.warn("Error setting sessionStorage key “user”:", error);
    }
  }, [user]);

  return (
    <UserContext.Provider value={{ role, setRole, user, setUser }}>
      {children}
    </UserContext.Provider>
  );
};
