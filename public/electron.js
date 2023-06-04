const path = require("path");
const url = require("url");

const { app, BrowserWindow, protocol, dialog, ipcMain } = require("electron");
const isDev = require("electron-is-dev");

const fs = require('fs');
let { PythonShell } = require('python-shell')
const unhandled = require('electron-unhandled');

const log = require('electron-log');

log.info('Hello, log');
log.warn('Some problem appears');

unhandled();



// Global variable
global.appRoot = path.resolve(__dirname)
var currData;
var modelArgs;
var win;

// // Conditionally include the dev tools installer to load React Dev Tools
// let installExtension, REACT_DEVELOPER_TOOLS; // NEW!

// if (isDev) {
//   const devTools = require("electron-devtools-installer");
//   installExtension = devTools.default;
//   REACT_DEVELOPER_TOOLS = devTools.REACT_DEVELOPER_TOOLS;
// } // NEW!

// Handle creating/removing shortcuts on Windows when installing/uninstalling
if (require("electron-squirrel-startup")) {
  app.quit();
} // NEW!
// test = hi


function createWindow() {
  // Create the browser window.
  win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      webSecurity: false,
      nodeIntegration: true,
      // contextIsolation: false,
      preload: path.join(__dirname, "preload.js")
    }
  });

  // and load the index.html of the app.
  // win.loadFile("index.html");
  win.loadURL(
    isDev
      ? "http://localhost:3000"
      : `file://${path.join(__dirname, "../build/index.html")}`
  );
  // .then(
  //   () => {window.webContents.send('sendModelJson')}
  // )
  // Open the DevTools.
  console.log("IS DEV?", isDev)
  if (isDev) {
    win.webContents.openDevTools();
  }

  return win;
}

/*
-------------------- IPC HANDLING -----------------------
*/
// Open file dialog
function handleFileOpen(e, message, win) {
  const dialogResult = dialog.showOpenDialog(win, {
    properties: ['openFile']
  })
  return dialogResult
  // console.log("hi")
  // const result = dialog.showOpenDialog({
  //   properties: ["openFile"],
  //   filters: [{ name: "Images", extensions: ["png","jpg","jpeg"] }]
  // });

  // result.then(({canceled, filePaths, bookmarks}) => {
  //   const base64 = fs.readFileSync(filePaths[0]).toString('base64');
  //   event.reply("chosenFile", base64);
  // });
  // return result
}

// Save file dialog
const ExcelExportData = 
    [{
      filename: 'img1.jpg', 
      majorgroup: 'Amphipoda', 
      individualcount: '2', 
      reviewed: '0'
    },
    {
      filename: 'img1.jpg', 
      majorgroup: 'Polychaeta', 
      individualcount: '3', 
      reviewed: '1'
    }]
function handleFileSave(e, win) {
  // console.log(projData)
  // console.log("HANDLE FILE SAVE WINDOW: ", win)
  const dialogResult = dialog.showSaveDialog(win, {
    defaultPath: path.join(__dirname, "../src/model_outputs"),
    properties: ['openFile'],
    extensions: ['json']
  })
  dialogResult.then(result => {
    console.log("RESULT: ", result)
    fs.writeFileSync(result.filePath, currData, 'utf-8');
  })
    return dialogResult
}

function handleDataSend(e, projData) {
  currData = projData
  console.log(currData)
}

async function handleAsyncMessage(event, arg) {
  console.log("Hi", arg)
}

async function handleCallScript(win) {
  const ret = await callScript(win)
  return ret
}

function handleArgs(e, python_args) {
  modelArgs = python_args
  console.log(modelArgs)
}

function callScript() {
  console.log("Running Python Script")
  // PythonShell.run('./src/pytest.py', null).then(messages=>{
  //   console.log(messages)
  //   console.log('finished');
  // });
  const options = {
    mode:'text',
    args: modelArgs
  }; 
  // PythonShell.run('./resources/app/model_core/modelExecutable.py', options).then(messages=>{ 
  PythonShell.run(path.join(__dirname, "../model_core/modelExecutable.py"), options).then(messages=>{
    console.log(messages)
    console.log("Finished python script")
    // console.log("WIN: ", win)
    win.webContents.send('finish-script', 1);

    var modelJsonPath = path.join(__dirname, "../src/model_outputs/model_output.json")
    var modelJsonFile = fs.readFileSync(modelJsonPath);
    console.log("JSON OBJECT: ", JSON.parse(modelJsonFile))
    win.webContents.send('sendModelJson', JSON.parse(modelJsonFile))
  });
  return 0;
}
// Call Python Script

function showImagePopup() {
  const options = {
    type: 'info',
    buttons: ['Close'],
    defaultId: 2,
    title: 'Warning',
    message: 'No next image',
  };
  const response = dialog.showMessageBox(null, options);
}

function showImagePopup2() {
  const options = {
    type: 'info',
    buttons: ['Close'],
    defaultId: 2,
    title: 'Warning',
    message: 'No previous image',
  };
  const response = dialog.showMessageBox(null, options);
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  ipcMain.handle('dialog:openFile', handleFileOpen);
  ipcMain.handle('async-message', handleAsyncMessage);
  ipcMain.handle('dialog:saveFile', handleFileSave);
  ipcMain.on('send-data', handleDataSend);
  ipcMain.on('call-python-file', callScript);
  ipcMain.on('next-image-popup', showImagePopup);
  ipcMain.on('prev-image-popup', showImagePopup2);
  ipcMain.on('send-args', handleArgs);
  ipcMain.handle('getPath', () => path.resolve(__dirname))
  
  ipcMain.on('ipc-example', async (event, arg) => {
    const msgTemplate = (pingPong) => `IPC test: ${pingPong}`;
    console.log(msgTemplate(arg));
    event.reply('ipc-example', msgTemplate('pong'));
  });

  const win = createWindow();




  // if (isDev) {
  //   installExtension(REACT_DEVELOPER_TOOLS)
  //     .then(name => console.log(`Added Extension:  ${name}`))
  //     .catch(error => console.log(`An error occurred: , ${error}`));
  // }
}); // UPDATED!

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.