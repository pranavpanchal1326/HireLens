import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import './index.css'
import { router } from './router.jsx'
import { initCursorLight } from './lib/atmosphere'

// Light the room — start the cursor-light engine (no-ops on touch / reduced-motion).
initCursorLight()

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
