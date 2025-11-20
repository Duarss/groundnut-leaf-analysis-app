// src/components/ui/Button.jsx
import React from "react";

const Button = ({ children, variant = "primary", ...rest }) => {
  return (
    <button className={`btn btn-${variant}`} {...rest}>
      {children}
    </button>
  );
};

export default Button;
