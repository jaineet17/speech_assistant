import './Footer.css';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer">
      <div className="container footer-container">
        <p className="copyright">
          &copy; {currentYear} Speech Assistant. All rights reserved.
        </p>
        <div className="footer-links">
          <a href="https://github.com/jaineet17/speech_assistant" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
          <a href="/privacy" className="footer-link">
            Privacy Policy
          </a>
          <a href="/terms" className="footer-link">
            Terms of Service
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 