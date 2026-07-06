import { createBrowserRouter, Navigate } from 'react-router-dom';
import Landing from './pages/Landing';
import SeekerLayout from './layouts/SeekerLayout';
import RecruiterLayout from './layouts/RecruiterLayout';
import Kit from './pages/Kit';
import Analyze from './pages/seeker/Analyze';
import Rescan from './pages/seeker/Rescan';
import Health from './pages/seeker/Health';
import Blind from './pages/seeker/Blind';
import Batch from './pages/recruiter/Batch';
import Ranked from './pages/recruiter/Ranked';
import Dashboard from './pages/recruiter/Dashboard';

// Route spine (§11). Two subtrees, two temperaments. Each layout owns its
// data-theme via its own chrome; screens fill in phase by phase (D4–D8). The
// placeholders keep every route reachable and on-brand until then.
export const router = createBrowserRouter([
  { path: '/', element: <Landing /> },
  { path: '/kit', element: <Kit /> },

  {
    path: '/seeker',
    element: <SeekerLayout />,
    children: [
      { index: true, element: <Navigate to="analyze" replace /> },
      { path: 'analyze', element: <Analyze /> },
      { path: 'rescan', element: <Rescan /> },
      { path: 'health', element: <Health /> },
      { path: 'blind', element: <Blind /> },
    ],
  },

  {
    path: '/recruiter',
    element: <RecruiterLayout />,
    children: [
      { index: true, element: <Navigate to="ranked" replace /> },
      { path: 'batch', element: <Batch /> },
      { path: 'ranked', element: <Ranked /> },
      { path: 'dashboard', element: <Dashboard /> },
    ],
  },

  { path: '*', element: <Navigate to="/" replace /> },
]);
