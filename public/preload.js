// All of the Node.js APIs are available in the preload process.
// It has the same sandbox as a Chrome extension.
const { contextBridge, ipcRenderer } = require("electron");

// As an example, here we use the exposeInMainWorld API to expose the browsers
// and node versions to the main window.
// They'll be accessible at "window.versions".
console.log("this is the preload script");

process.once("loaded", () => {
  contextBridge.exposeInMainWorld("versions", process.versions);
  contextBridge.exposeInMainWorld('electronAPI', {
    ipcR: {
      myPing() {
        ipcRenderer.send('ipc-example', 'ping');
      },
      openFile: () => ipcRenderer.invoke('dialog:openFile'),
      saveFile: () => ipcRenderer.invoke('dialog:saveFile'),
      sendProjectData: (projData) => ipcRenderer.send('send-data', projData),
      asyncMessage: () => ipcRenderer.send('async-message'),
      callPythonFile: () => ipcRenderer.send('call-python-file'),
      nextImagePopup: () => ipcRenderer.send('next-image-popup'),
      prevImagePopup: () => ipcRenderer.send('prev-image-popup'),
      handleScriptFinish: (callback) => ipcRenderer.on('finish-script', callback),
      sendPythonArgs: (pythonArgs) => ipcRenderer.send('send-args', pythonArgs),
      receiveRoot: (dir_name) => ipcRenderer.on('send-root', dir_name),
      getPath: () => ipcRenderer.invoke('getPath'),
      sendModelJson: (modelJson) => {
        ipcRenderer.on('sendModelJson', modelJson)
      },
      openProjectData: () => ipcRenderer.send('openProject'), // used in ExistingProject.js: creates ipc call to use fs to choose an existing project

      once(channel, func) {
        const validChannels = ['ipc-example'];
        if (validChannels.includes(channel)) {
          // Deliberately strip event as it includes `sender`
          ipcRenderer.once(channel, (event, ...args) => func(...args));
        }
      },
      // on(channel) {
      //   const validChannels = ['dialog:openFile', 'OPEN_FILE_PATH'];
      //   if (validChannels.includes(channel)) {
      //     ipcRenderer.on(channel, (event, result) => {
      //       const src = `data:image/jpg;base64,${result}`
      //       return src;
      //     })
      //   }
      // }
    }
  })
});

