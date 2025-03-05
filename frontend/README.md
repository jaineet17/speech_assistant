# Enhanced Speech Assistant Frontend

This directory contains the React/TypeScript frontend for the Enhanced Speech Assistant project. The frontend provides a modern, responsive user interface for interacting with the speech assistant through voice commands.

## Features

- Voice recording and playback
- Real-time transcription display
- AI assistant interaction
- Performance metrics visualization
- Responsive design for desktop and mobile

## Technology Stack

- React 18
- TypeScript
- Vite (build tool)
- React Router (navigation)
- CSS Modules (styling)

## Setup

1. Make sure you have Node.js (v16+) and npm installed
2. Install dependencies:

```bash
npm install
```

## Development

To start the development server:

```bash
npm run dev
```

This will start the Vite development server, typically on http://localhost:5173.

## Building for Production

To build the frontend for production:

```bash
npm run build
```

This will create optimized production files in the `dist` directory.

## Project Structure

```
frontend/
├── public/          # Static assets
├── src/             # Source code
│   ├── components/  # Reusable UI components
│   ├── pages/       # Page components
│   ├── services/    # API services
│   ├── styles/      # Global styles
│   ├── App.tsx      # Main application component
│   └── main.tsx     # Application entry point
├── index.html       # HTML template
├── tsconfig.json    # TypeScript configuration
└── vite.config.ts   # Vite configuration
```

## API Integration

The frontend communicates with the backend API running on http://localhost:5050. The API endpoints used include:

- `/api/process-audio` - Process audio recordings
- `/api/process-text` - Process text input
- `/api/history` - Get transcription history
- `/api/status` - Get system status

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build locally

## Browser Compatibility

The application is designed to work with modern browsers that support the MediaRecorder API, including:

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest) 