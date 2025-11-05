import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const toastConfig = {
  position: "top-center",
  autoClose: 3000,
  hideProgressBar: true,
  closeOnClick: true,
  pauseOnHover: false,
  draggable: false,
  progress: undefined,
};

function Toaster() {
  return (
    <ToastContainer
      position="top-center"
      autoClose={3000}
      limit={1}
      hideProgressBar
      newestOnTop
      closeOnClick
      rtl={false}
      pauseOnFocusLoss={false}
      draggable={false}
      pauseOnHover={false}
      theme="light"
      className="!top-4"
    />
  );
}

const customToast = {
  success: (message) => {
    toast.dismiss();
    toast.success(message, toastConfig);
  },
  error: (message) => {
    toast.dismiss();
    toast.error(message, toastConfig);
  },
  info: (message) => {
    toast.dismiss();
    toast.info(message, toastConfig);
  },
  warning: (message) => {
    toast.dismiss();
    toast.warning(message, toastConfig);
  }
};

export { Toaster, customToast as toast };
