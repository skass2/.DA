// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut } from "firebase/auth";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyD9dALi9ID5RgvLG56ZAGu99Y5Vb1GqVsA",
  authDomain: "mychatbot-7021.firebaseapp.com",
  projectId: "mychatbot-7021",
  storageBucket: "mychatbot-7021.firebasestorage.app",
  messagingSenderId: "385684644828",
  appId: "1:385684644828:web:e5a7da2ad5c2a951312519",
  measurementId: "G-PMNC6QCTCB"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const analytics = getAnalytics(app);

// Khởi tạo Authentication
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
export { signInWithPopup, signOut };