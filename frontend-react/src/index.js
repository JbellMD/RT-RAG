import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ChakraProvider, extendTheme } from '@chakra-ui/react';

// Optional: Define a custom theme or extend the default
// If you want to use the default theme, you can omit the theme prop 
// or pass an empty extendTheme call: theme={extendTheme({})}.
const theme = extendTheme({
  // Example: Customizing colors (optional)
  // colors: {
  //   brand: {
  //     900: '#1a365d',
  //     800: '#153e75',
  //     700: '#2a69ac',
  //   },
  // },
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ChakraProvider theme={theme}>
      <App />
    </ChakraProvider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
