// src/components/ui/Card.jsx
import React from "react";

const Card = ({ title, subtitle, children, footer }) => {
  return (
    <div className="card">
      {(title || subtitle) && (
        <div className="card-header">
          {title && <h3>{title}</h3>}
          {subtitle && <p className="card-subtitle">{subtitle}</p>}
        </div>
      )}
      <div className="card-body">{children}</div>
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
};

export default Card;
