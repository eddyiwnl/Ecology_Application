import { Link } from "react-router-dom";
import React, { useEffect, useRef, useState } from "react";
import Dropdown from 'react-bootstrap/Dropdown'
import DropdownButton from 'react-bootstrap/DropdownButton'
import './ModelOutput.css'
import XLSX from 'sheetjs-style';
import * as FileSaver from 'file-saver';
// import log from 'electron-log/renderer';

// log.info('Log from ModelOutput.js')

// const unhandled = require('electron-unhandled');

// unhandled();


// const PureCanvas = React.forwardRef((props, ref) => <canvas ref={ref} />)

/* TODO
One of two functionalities: 
*/
const ModelOutput = ({projectData, setProjectData, fileName}) => {
    // sessionStorage.clear();

    //---------------------------------------------Initializing variables------------------------------------------
    const [outputGroup, setOutputGroup] = useState("") // test output
    const [currElement, setCurrElement] = useState(0) // current box id
    const [show, setShow] = useState(false) // show or no show button
    const [showDropDown, setShowDropDown] = useState(false) // show or no show drop-down menu during edit
    const [deleteButton, setDeleteButton] = useState(false) // show or no show the delete button
    const [bboxs, setBboxs] = useState([]) // bbox_list
    const [currCtx, setCurrCtx] = useState() // current context (usually the canvasRef.current)
    const [currMajorGroup, setCurrMajorGroup] = useState("") // current selected major group
    const [confidence, setConfidence] = useState(0) // current selected confidence
    const [numElements, setNumElements] = useState(0) // number of bounding boxes
    const [currFilepath, setCurrFilepath] = useState("")
    const [fileChange, setFileChange] = useState(false)
    const [selectedImage, setSelectedImage] = useState("")
    // const [hover, setHover] = useState(false)

    const [inDelete, setInDelete] = useState(false); // whether or not the user just deleted a box
    const [inEdit, setInEdit] = useState(false) // whether or not the user is currently editing


    const canvasRef = useRef();
    const textCanvasRef = useRef();

    var bbox_list = []
    var img_dir_name_map = []

    var closeEnough = 5;
    var dragTL = false;
    var dragBL = false;
    var dragTR = false;
    var dragBR = false;
    var dragBox = false;
    var hover = false;
    var id;
    // var isDeleted = false;

    var mDownX = 0; // X position of mouseDown
    var mDownY = 0; // Y position of mouseDown
    // var inEdit = false;


    // Hash table: major group -> bbox color
    /* 
    Amphipoda: Red
    Bivalvia: Yellow
    Cumacea: Dark Green
    Gastropoda: Magenta
    Insecta: Light Purple
    Ostracoda: Lime Green
    Polychaeta: Light Blue
    Other Annelida: White
    Other Crustacea: Light Red
    Other: Grey
    */
    const major_group_color = new Map();
    major_group_color.set("Amphipoda", "#E52D00")
    major_group_color.set("Bivalvia", "#FFED0E")
    major_group_color.set("Cumacea", "#FFBB00") 
    major_group_color.set("Gastropoda", "#FC30F6")
    major_group_color.set("Insecta", "#A9A0FF")
    major_group_color.set("Ostracoda", "#2DFF29")
    major_group_color.set("Polychaeta", "#2FCAF4")
    major_group_color.set("Other", "#1803FC")
    major_group_color.set("Other Annelida", "#38B2A7")
    major_group_color.set("Other Crustacea", "#FFBBBB")
    major_group_color.set("Unknown", "#7803FC")
    major_group_color.set("Custom Label", "#8C8B8B")


    

    //---------------------------------------------Initializing JSON / images------------------------------------------
    var root_path;
    var new_root_path;

    window.electronAPI.ipcR.getPath()
    .then((appDataPath) => {
        root_path = appDataPath
        new_root_path = JSON.parse(JSON.stringify(root_path).replaceAll('\\\\', '/'))
        // console.log(new_root_path)
    })
    // ORDER IS IMPORTANT: LOAD JSON THEN IF JSON IS NOT AVAILABLE, WE KNOW ITS A NEW PROJECT AND THEN LOAD FILELIST
    // window.electronAPI.ipcR.sendModelJson((event, modelJsonFile) => {
    //     console.log("SENT MODEL JSON FILE FROM MAIN: ", modelJsonFile)
    // })
    // var testJson = require('./resources/app/src/model_outputs/model_output.json');
    var testJson = JSON.parse(sessionStorage.getItem("init-model"));
    // console.log("THE TESTJSON IS: ", testJson)
    // var testJson = require('./model_outputs/model_output.json');
    var fileList;

    var editJson;
    editJson = JSON.parse(sessionStorage.getItem("curr_json"));
    if(!editJson) {
        console.log("Empty storage, loading JSON")
        editJson = JSON.parse(JSON.stringify(testJson))
        fileList = JSON.parse(sessionStorage.getItem("fileList"))
    }
    else {
        fileList = []
        for (var key in editJson) {
            if (editJson.hasOwnProperty(key)) {
                fileList.push(key)
                // console.log(key + " -> " + testing[key]);
            }
        }
    }
    console.log("------------------------------------------------------")
    console.log(editJson)


    var currImageId = sessionStorage.getItem("curr_image_id");
    if(!currImageId) {
        currImageId = 0;
        sessionStorage.setItem("curr_image_id", currImageId);
    }
    else {
        currImageId = parseInt(currImageId)
    }
    // const fileList = JSON.parse(sessionStorage.getItem("fileList"))
    // console.log("passing files:", fileList)
    // console.log("passing files first file path:", fileList[currImageId])

    //need to change back slashes to forward slashes, insert double backslash in front of spaces, and add file:///
    const correctFilepaths = []
    for(var i = 0; i < fileList.length; i++) {
        const newpath = fileList[i].replaceAll("\\", "/")
        const newpath2 = newpath.replaceAll(" ", "\\ ")
        const finalpath = "file:///" + newpath2
        correctFilepaths.push(finalpath)
    }

    // console.log(correctFilepaths)
    // console.log(correctFilepaths[0])
    const genFilePath = (filePaths) => {
        const fixedPath = correctFilepaths[currImageId].slice(8).replaceAll("\\ ", " ")
        return fixedPath
    }
    const currImage = genFilePath(correctFilepaths)
    console.log("Current image: ", currImage)

    var split_json;
    var temp_id = 0; 
    Object.keys(editJson).forEach(function(key) {
        split_json = key.split('/')
        img_dir_name_map.push({file: split_json[split_json.length - 1], path: key, id: temp_id})
        // console.log(img_dir_name_map)
        temp_id++;
    });

    // const imgMapList = Object.keys(img_dir_name_map).map(key => ({
    //     path: img_dir_name_map[key]
    // }))
    // console.log("IMG MAP LIST: ", imgMapList)
    // imgMapList.map((country, key) => {
    //     console.log("country: ", country)
    //     console.log("key: ", key)
    // })
    // imgMapList.map((country, key) => (
    //     console.log("country: ", country)
    //     console.log("key: ", key)
    // ))

    // var currImage = "M12_2_Apr19_3.jpg";

    // setProjectData(testJson)

    //---------------------------------------------------Download to excel code------------------------------------------
    
    //creating json data from model

    // const subgroups = [];

    const exportToExcel = async (fileName, editJson) => {

        const newdict = {}
        console.log(editJson)

        function containsObject(obj, list) {
            var i;
            for (i = 0; i < Object.keys(list).length; i++) {
                if (Object.keys(list)[i] == obj) {
                    return true;
                }
            }
        
            return false;
        }

        
        for (let i = 0; i < Object.keys(editJson).length; i++) {
            newdict[Object.keys(editJson)[i]] = {}
            for (let j = 0; j < editJson[Object.keys(editJson)[i]].predictions.pred_labels.length; j++){
                if(containsObject(editJson[Object.keys(editJson)[i]].predictions.pred_labels[j], newdict[Object.keys(editJson)[i]])){
                    newdict[Object.keys(editJson)[i]][editJson[Object.keys(editJson)[i]].predictions.pred_labels[j]]++;
                } else {
                    newdict[Object.keys(editJson)[i]][editJson[Object.keys(editJson)[i]].predictions.pred_labels[j]] = 1;
                }
            }
        }

        //restrucute dictionary so that it matches desired format
        const finalData = []

        for (let i = 0; i < Object.keys(newdict).length; i++) { //for every 7 images
            //console.log(Object.keys(newdict))
            //loop through dictionary of image
            for (let j = 0; j < Object.keys(newdict[Object.keys(newdict)[i]]).length; j++) {
                //console.log(newdict[Object.keys(newdict)[i]])
                const finalDict = {}
                finalDict['File Name'] = Object.keys(newdict)[i]
                finalDict['Major Group'] = Object.keys(newdict[Object.keys(newdict)[i]])[j]
                finalDict['Individual Count'] = Object.values(newdict[Object.keys(newdict)[i]])[j]
                finalDict['Manually Reviewed'] = 0
                // finalDict['Additional Label'] = subgroups[0] //will have to change this once we have multiple images
                finalData.push(finalDict)
            }
        }

        //loop through major groups of old dictionary to get total counts 
        const finalCounts = []
        const countsDict = {}

        for (let i = 0; i < Object.keys(newdict).length; i++) {
            for (let j = 0; j < Object.keys(newdict[Object.keys(newdict)[i]]).length; j++) {
                if(containsObject(Object.keys(newdict[Object.keys(newdict)[i]])[j], countsDict)){
                    countsDict[Object.keys(newdict[Object.keys(newdict)[i]])[j]]++;
                } else {
                    countsDict[Object.keys(newdict[Object.keys(newdict)[i]])[j]] = 1;
                }
            }
        }

        finalCounts.push(countsDict)


        const fileType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;charset=UTF-8';
        const fileExtension = '.xlsx';

        const ws1 = XLSX.utils.json_to_sheet(finalData);
        const ws2 = XLSX.utils.json_to_sheet(finalCounts);
        const wb = { Sheets: { 'Results': ws1, 'Total Counts': ws2 }, SheetNames: ['Results', 'Total Counts'] };
        const excelBuffer = XLSX.write(wb, {bookType: 'xlsx', type: 'array' });
        const Exceldata = new Blob([excelBuffer], { type: fileType });
        FileSaver.saveAs(Exceldata, fileName + fileExtension);
    }
    //-----------------------------------------------End of Download to excel code-----------------------------------------

    const setCanvasDims = (ctx) => {
        const canvas = ctx.canvas
        canvas.width = 825;
        canvas.height = 550;
        console.log("canvas width: ", canvas.width)
        console.log("canvas height: ", canvas.height)
    }
    const drawBBox = (ctx, bbox, labels, scores) => {
        const x1 = bbox[0]
        const y1 = bbox[1]
        const x2 = bbox[2]
        const y2 = bbox[3]
        // console.log(bbox)
        ctx.strokeStyle = major_group_color.get(labels);
        ctx.fillStyle = major_group_color.get(labels);
        ctx.globalAlpha = 0.25;
        ctx.lineWidth = 2;
        // strokeRect(x, y, width, height)
        // 6.545 is our scaling from the original image dimensions (5400px x 3600px): we scale it down to (825px x 550px)
        ctx.strokeRect(x1/6.545, y1/6.545, (x2-x1)/6.545, (y2-y1)/6.545);
        ctx.fillRect(x1/6.545, y1/6.545, (x2-x1)/6.545, (y2-y1)/6.545);

        writeText(ctx, { text: labels+": "+Math.round(scores*100)/100, x: x1/6.545, y: y1/6.545 });

        
        // bbox_list.push({x: x1/6.545, y: y1/6.545, w: (x2-x1)/6.545, h: (y2-y1)/6.545, color: major_group_color.get(labels), majorgroup: labels})
        // setBboxs(bbox_list)
        // ctx.clearRect((x1/6.545)-3, (y1/6.545)-3, ((x2-x1)/6.545)+4, ((y2-y1)/6.545)+4) 
        // -3 because lineWidth is creating a border outside the rect pixels
        // +4 is because the previous line doesnt reach the last line

    };
    const updateBBox = (ctx, bbox, labels, scores) => {
        const x1 = bbox[0]
        const y1 = bbox[1]
        const x2 = bbox[2]
        const y2 = bbox[3]        
        bbox_list.push({x: x1/6.545, y: y1/6.545, w: (x2-x1)/6.545, h: (y2-y1)/6.545, color: major_group_color.get(labels), majorgroup: labels, score: Math.round(scores*100)/100})
        // console.log("BBOX LIST: ", bbox_list)
        setBboxs(bbox_list)
        // ctx.clearRect((x1/6.545)-3, (y1/6.545)-3, ((x2-x1)/6.545)+4, ((y2-y1)/6.545)+4) 
        // -3 because lineWidth is creating a border outside the rect pixels
        // +4 is because the previous line doesnt reach the last line

    };

    const writeText = (ctx, info, style = {}) => {
        // ctx.clearRect(0,0,1000,1000);
        const { text, x, y } = info
        const { fontSize = 20, fontFamily = 'Arial', color = "white", textAlign = 'left', textBaseline = 'top' } = style;
        ctx.save();
        ctx.globalAlpha=1.0;
        ctx.beginPath();
        ctx.font = fontSize + 'px ' + fontFamily;
        ctx.textAlign = textAlign;
        ctx.textBaseline = textBaseline;
        ctx.fillStyle = color;
        ctx.fillText(text, x, y);
        ctx.stroke();
        ctx.fill();
        ctx.restore();
    }
    
    useEffect(() => {
        // TODO: LOOK AT ALL THE LOCATIONS WHERE I USE 'currElement' AND CHANGE PROPERLY and change to id if necessary, 
        // TODO: work on dragging bbox functionality
        console.log("USE EFFECT 2");
        console.log("BBOXS: ", bboxs)
        const ctx = canvasRef.current.getContext("2d")
        
        for (var i = 0; i < bboxs.length; i++) {
            if((i == currElement) && inEdit) {
                ctx.strokeStyle = "white";
                ctx.fillStyle = "white";
            }
            else {
                ctx.strokeStyle = bboxs[i].color;
                ctx.fillStyle = bboxs[i].color;
            }
            ctx.globalAlpha = 0.25;
            ctx.lineWidth = 2;
            // strokeRect(x, y, width, height)
            // 6.545 is our scaling from the original image dimensions (5400px x 3600px): we scale it down to (825px x 550px)
            ctx.strokeRect(bboxs[i].x, bboxs[i].y, bboxs[i].w, bboxs[i].h);
            ctx.fillRect(bboxs[i].x, bboxs[i].y, bboxs[i].w, bboxs[i].h);

            writeText(ctx, { text: bboxs[i].majorgroup+": "+bboxs[i].score, x: bboxs[i].x, y: bboxs[i].y });

            
        }
        // for (var i = 0; i < testJson["M12_2_Apr19_3.jpg"].truth.true_boxes.length; i++)
        // {
        //     if(i == 2) {
        //         continue
        //     }
        //     console.log(i)
        //     var true_labels = testJson["M12_2_Apr19_3.jpg"].truth.true_labels
        //     var true_bbox = testJson["M12_2_Apr19_3.jpg"].truth.true_boxes
        //     drawBBox(ctx, true_bbox[i], true_labels[i])
        // }
        // console.log(bbox_list)

        // var id;
        var clicked = false;
        var dragging = false;
        var _i, _b;

        function checkCloseEnough(p1, p2) {
            console.log("CLose enough: ", closeEnough)
            return Math.abs(p1 - p2) < closeEnough;
        }
        canvasRef.current.onmousedown = function(e) {
            var r = canvasRef.current.getBoundingClientRect(),
                x = e.clientX - r.left, y = e.clientY - r.top;
            mDownX = x;
            console.log("MOUSE DOWN X: ", mDownX)
            mDownY = y;
            hover = false;

            console.log("INEDIT MOUSE DOWN")

            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

            for(var i = bboxs.length - 1, b; b = bboxs[i]; i--) {
                if(x >= b.x - closeEnough && x <= b.x + b.w + closeEnough &&
                y >= b.y - closeEnough && y <= b.y + b.h + closeEnough) {
                    // The mouse honestly hits the rect
                    hover = true;
                    id = i;
                    setCurrElement(id)
                    setShow(true);
                    setInEdit(true);
                    setCurrMajorGroup(b.majorgroup);
                    setConfidence(b.score);
                    break;
                }
                else{
                    hover = false;
                    id = -1
                    setCurrMajorGroup('');
                    setConfidence(0);
                    setShow(false);
                    setInEdit(false);
                }
            }
            console.log('coords: ', x, y)
            console.log("CURRENT ELEMENT: ", currElement)
            console.log("ID: ", id)
            // console.log(id)
            // if (id == -1) {
            //     dragging = false;
            //     console.log("None")
            // }
            if(inEdit) {
                console.log("YES")
                console.log("THE ID IS: ", id)
                
                // 6. none of them
                setShow(true);
                if (id == -1) {
                    dragging = false;
                    console.log("None")
                    ctx.fillStyle = "white";
                    ctx.fillRect(bboxs[currElement].x, bboxs[currElement].y, bboxs[currElement].w, bboxs[currElement].h);
                }
                // else if(!id) {
                //     dragging = false;
                //     console.log("None")
                // }
                // 1. top left
                else if (checkCloseEnough(x, bboxs[id].x) && checkCloseEnough(y, bboxs[id].y)) {
                    dragging = true;
                    dragTL = true;
                    console.log("Dragging top left")
                }
                // 2. top right
                else if (checkCloseEnough(x, bboxs[id].x + bboxs[id].w) && checkCloseEnough(y, bboxs[id].y)) {
                    dragging = true;
                    dragTR = true;
                    console.log("Dragging top right")
                }
                // 3. bottom left
                else if (checkCloseEnough(x, bboxs[id].x) && checkCloseEnough(y, bboxs[id].y + bboxs[id].h)) {
                    dragging = true;
                    dragBL = true;
                    console.log("Dragging bottom left")
                }  
                // 4. bottom right
                else if (checkCloseEnough(x, bboxs[id].x + bboxs[id].w) && checkCloseEnough(y, bboxs[id].y + bboxs[id].h)) {
                    dragging = true;
                    dragBR = true;
                    console.log("Dragging bottom right")
                }
                // 5. dragging the box itself
                else if(id != -1) {
                    dragging = true;
                    dragBox = true;
                    console.log("Dragging box")
                }
                // 6. none of them
                else {
                    dragging = false;
                    console.log("None")
                }
            } else {
                // do nothing
            }

            // ctx.clearRect(bbox_list[id].x, bbox_list[id].y, bbox_list[id].w, bbox_list[id].h);

            for(_i = 0; _b = bboxs[_i]; _i++) {
                if(hover && id === _i) {
                    setCurrElement(_i)
                    setShow(true);
                    ctx.fillStyle = "white";
                }
                else {
                    ctx.fillStyle = _b.color;
                }
                // ctx.fillStyle = (hover && id === _i) ? "red" : _b.color;
                // ctx.clearRect(_b.x, _b.y, _b.w, _b.h);
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                ctx.fillRect(_b.x, _b.y, _b.w, _b.h);
                writeText(ctx, { text: _b.majorgroup+": "+_b.score, x: _b.x, y: _b.y });

                // setOutputGroup(_b.majorgroup);
            }
            // renderMap2(x, y);
        }
        
        canvasRef.current.onmouseup = function(e) {
            dragTL = dragBR = dragTR = dragBL = dragBox = false;
            dragging = false;
            // ctx.clearRect(bbox_list[id].x, bbox_list[id].y, bbox_list[id].w, bbox_list[id].h);
            // if(dragging) {
            //     drawAnchors();
            // }

            if(inEdit) {
                console.log("MOUSE UP INEDIT TRUE")
                for(_i = 0; _b = bboxs[_i]; _i ++) {
                    if(hover && id === _i) {
                        console.log("ID ", id, "IS WHITE")
                        ctx.fillStyle = "white";
                        ctx.clearRect(_b.x, _b.y, _b.w, _b.h);
                        ctx.fillRect(_b.x, _b.y, _b.w, _b.h);
                    }
                    // ctx.fillStyle = (hover && id === _i) ? "red" : _b.color;
                    // setOutputGroup(_b.majorgroup);
                }
            }
            else {
                console.log("MOUSE UP INEDIT FALSE")
                if(id == -1) {
                    for(_i = 0; _b = bboxs[_i]; _i ++) {
                        ctx.fillStyle = _b.color;
                        // ctx.fillStyle = (hover && id === _i) ? "red" : _b.color;
                        ctx.clearRect(_b.x, _b.y, _b.w, _b.h);
                        ctx.fillRect(_b.x, _b.y, _b.w, _b.h);
                        writeText(ctx, { text: _b.majorgroup+": "+_b.score, x: _b.x, y: _b.y });

                        // setOutputGroup(_b.majorgroup);
                    }
                }
            }
        }

        canvasRef.current.onmousemove = function(e) {
            var r = canvasRef.current.getBoundingClientRect(),
                x = e.clientX - r.left, y = e.clientY - r.top;
            // console.log("MDOWN X: ", mDownX)
            // console.log("CURR X: ", x)
            // console.log('onmousemove id', id)
            // console.log("inedit: ", inEdit)
            if(inEdit) {
                drawAnchors();
                console.log("CURRELEMENT: ", currElement)

                // Clear all rects if not the one in edit
                for(_i = 0; _b = bboxs[_i]; _i ++) {
                    if(currElement != _i) {
                        // ctx.clearRect(_b.x, _b.y, _b.w, _b.h)
                        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                        // console.log("DRAWING")
                        drawAnchors();
                        // writeText(ctx, { text: _b.majorgroup, x: _b.x, y: _b.y });

                    }
                    else {
                        ctx.fillStyle = "white";
                        // ctx.clearRect(_b.x, _b.y, _b.w, _b.h);
                        ctx.fillRect(_b.x, _b.y, _b.w, _b.h);
                        // console.log("WRITING")
                        writeText(ctx, { text: _b.majorgroup, x: _b.x, y: _b.y });

                    }

                    // ctx.fillStyle = (hover && id === _i) ? "red" : _b.color;
                    // setOutputGroup(_b.majorgroup);
                }
                if (dragTL) {
                    setShow(true);
                    bboxs[currElement].w += bboxs[currElement].x - x;
                    bboxs[currElement].h += bboxs[currElement].y - y;
                    bboxs[currElement].x = x;
                    bboxs[currElement].y = y;
                } else if (dragTR) {
                    setShow(true);
                    bboxs[currElement].w = Math.abs(bboxs[currElement].x - x);
                    bboxs[currElement].h += bboxs[currElement].y - y;
                    bboxs[currElement].y = y;
                } else if (dragBL) {
                    setShow(true);
                    bboxs[currElement].w += bboxs[currElement].x - x;
                    bboxs[currElement].h = Math.abs(bboxs[currElement].y - y);
                    bboxs[currElement].x = x;
                } else if (dragBR) {
                    setShow(true);
                    bboxs[currElement].w = Math.abs(bboxs[currElement].x - x);
                    bboxs[currElement].h = Math.abs(bboxs[currElement].y - y);
                } else if (dragBox) {
                    setShow(true);
                    var dx = x - mDownX;
                    var dy = y - mDownY;
                    mDownX = x;
                    mDownY = y;
                    bboxs[currElement].x += dx;
                    bboxs[currElement].y += dy;
                }
                
                // ctx.clearRect(bbox_list[id].x, bbox_list[id].y, bbox_list[id].w, bbox_list[id].h);
                if(dragging) {
                    setShow(true);
                    // inEdit=true;
                    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                    drawAnchors();
                }
                else {
                    ctx.clearRect(bboxs[currElement].x, bboxs[currElement].y, bboxs[currElement].w, bboxs[currElement].h);
                }
                draw(true);
                // console.log("inEdit useEffect: ", inEdit)
            }
            else {
                // ctx.clearRect(bbox_list[id].x, bbox_list[id].y, bbox_list[id].w, bbox_list[id].h);
                if(dragging) {
                    setShow(true);
                    // inEdit=true;
                    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                }
                else {
                    for(_i = 0; _b = bboxs[_i]; _i ++) {

                        // writeText(ctx, { text: _b.majorgroup+": "+_b.score, x: _b.x, y: _b.y });

                        // if(hover && currElement === _i) {
                        //     ctx.fillStyle = "white";
                        //     ctx.clearRect(_b.x, _b.y, _b.w, _b.h)
                        //     ctx.fillRect(_b.x, _b.y, _b.w, _b.h);
                        // }
                        // ctx.fillStyle = (hover && id === _i) ? "red" : _b.color;
                        // setOutputGroup(_b.majorgroup);
                    }
                    // ctx.clearRect(bboxs[id].x, bboxs[id].y, bboxs[id].w, bboxs[id].h);
                }
                // console.log("HOVER: ", hover)
                // draw(hover);
                // console.log("inEdit useEffect: ", inEdit)
            }
        }

        function draw(isHover) {
            // console.log("curr: ", currElement)
            // console.log("Draw id: ", id)
            // console.log("HOVER: ", hover)
            if(isHover == true) {
                ctx.fillStyle = "white";
            } else {
                ctx.fillStyle = bboxs[currElement].color
            }
            ctx.globalAlpha = 0.25;
            ctx.clearRect(bboxs[currElement].x, bboxs[currElement].y, bboxs[currElement].w, bboxs[currElement].h)
            ctx.fillRect(bboxs[currElement].x, bboxs[currElement].y, bboxs[currElement].w, bboxs[currElement].h)
            // drawAnchors();
        }

        function singleAnchor(x, y, radius) {
            ctx.fillStyle = "#FFFFFF";
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
        }

        function clearAnchor(x, y, radius) {
            ctx.globalCompositionOperation = 'destination-out'
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
        }
    
        function drawAnchors() {
            singleAnchor(bboxs[currElement].x, bboxs[currElement].y, closeEnough) // top left
            singleAnchor(bboxs[currElement].x + bboxs[currElement].w, bboxs[currElement].y, closeEnough) // top right
            singleAnchor(bboxs[currElement].x, bboxs[currElement].y + bboxs[currElement].h, closeEnough) // bottom left
            singleAnchor(bboxs[currElement].x + bboxs[currElement].w, bboxs[currElement].y + bboxs[currElement].h, closeEnough) // bottom right

            // clearAnchor(bbox_list[id].x, bbox_list[id].y, closeEnough) // top left
            // clearAnchor(bbox_list[id].x + bbox_list[id].w, bbox_list[id].y, closeEnough) // top right
            // clearAnchor(bbox_list[id].x, bbox_list[id].y + bbox_list[id].h, closeEnough) // bottom left
            // clearAnchor(bbox_list[id].x + bbox_list[id].w, bbox_list[id].y + bbox_list[id].h, closeEnough) // bottom right
        }

    }, [inEdit]);

    useEffect(() => {
        console.log("FIRING")
        const ctx = canvasRef.current.getContext("2d")
        const text_ctx = textCanvasRef.current.getContext("2d")
        setCanvasDims(ctx);
        setCanvasDims(text_ctx);
        // setInEdit(true);
        setCurrCtx(ctx)

        console.log("testJSON: ", testJson)
        // Using truth
        // for (var i = 0; i < testJson["M12_2_Apr19_3.jpg"].truth.true_boxes.length; i++)
        // {
        //     if(i == 2) {
        //         continue
        //     }
        //     console.log(i)
        //     var true_labels = testJson["M12_2_Apr19_3.jpg"].truth.true_labels
        //     var true_bbox = testJson["M12_2_Apr19_3.jpg"].truth.true_boxes
        //     drawBBox(ctx, true_bbox[i], true_labels[i])
        //     updateBBox(ctx, true_bbox[i], true_labels[i])
        // }
        // console.log(editJson["C:/Users/edwar/Desktop/Cal Poly/Ecology Project/forge-test-2/src/photos_src/M12_2_Oct19_2.jpg"])
        console.log("EDITJSON IS: ", editJson)
        for (var key2 in editJson) {
            if (editJson.hasOwnProperty(key2)) {
                console.log(key2 + " -> " + editJson[key2]);
            }
        }
        console.log("CURR IMAGE IS: ", currImage)
        editJson[currImage].predictions.area = []

        for (var i = 0; i < editJson[currImage].predictions.pred_boxes.length; i++)
        {
            // numbers go x1, y1, x2, y2 (area = (x2 - x1) * (y2 - y1 ))
            var current_box = editJson[currImage].predictions.pred_boxes[i]
            //add area as a key in the JSON data
            var area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            editJson[currImage].predictions.area[i] = area
            
        }
        
        var area_items = Object.keys(editJson[currImage].predictions.area).map(
            (key) => { return [key, editJson[currImage].predictions.area[key]] });

        
        area_items.sort((first, second) => { return first[1] - second[1] });

        var sorted_json = area_items.map(
            (e) => { return e[0] }).reverse();

        console.log("SORTED: ", sorted_json)

        // Using predictions
        for (var sorter_index = 0; sorter_index < sorted_json.length; sorter_index++)
        {
            // console.log(i)
            var pred_labels = editJson[currImage].predictions.pred_labels
            var pred_bbox = editJson[currImage].predictions.pred_boxes
            var pred_scores = editJson[currImage].predictions.pred_scores
            console.log("PRED BOXES: ", pred_bbox)
            drawBBox(ctx, pred_bbox[sorted_json[sorter_index]], pred_labels[sorted_json[sorter_index]], pred_scores[sorted_json[sorter_index]])
            updateBBox(ctx, pred_bbox[sorted_json[sorter_index]], pred_labels[sorted_json[sorter_index]], pred_scores[sorted_json[sorter_index]])
        }
        console.log(bbox_list)

        var hover = false, id;
        var clicked = false;
        var dragging = false;
        var _i, _b;

        function checkCloseEnough(p1, p2) {
            return Math.abs(p1 - p2) < closeEnough;
        }
        // canvasRef.current.onmousemove = function(e) {
        //     // Get the current mouse position
        //     var r = canvasRef.current.getBoundingClientRect(),
        //         x = e.clientX - r.left, y = e.clientY - r.top;
        //     hover = false;

        //     ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        //     for(var i = bbox_list.length - 1, b; b = bbox_list[i]; i--) {
        //         if(x >= b.x && x <= b.x + b.w &&
        //         y >= b.y && y <= b.y + b.h) {
        //             // The mouse honestly hits the rect
        //             hover = true;
        //             id = i;
        //             // canvas.addEventListener
        //             break;
        //         }
        //     }
        //     // Draw the rectangles by Z (ASC)
        //     renderMap();
        // }

        canvasRef.current.onmousedown = function(e) {
            var r = canvasRef.current.getBoundingClientRect(),
                x = e.clientX - r.left, y = e.clientY - r.top;
            hover = false;

            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

            for(var i = bbox_list.length - 1, b; b = bbox_list[i]; i--) {
                if(x >= b.x && x <= b.x + b.w &&
                y >= b.y && y <= b.y + b.h) {
                    // The mouse honestly hits the rect
                    hover = true;
                    id = i;
                    setShow(true);
                    setInEdit(true);
                    setCurrMajorGroup(b.majorgroup);
                    setConfidence(b.score);
                    break;
                }
                else{
                    hover = false;
                    setCurrMajorGroup('');
                    setConfidence(0);
                    setShow(false);
                    setInEdit(false);
                }
            }
            console.log('coords: ', x, y)
            console.log("ID: ", id)
            // console.log(id)
            if (checkCloseEnough(x, bbox_list[id].x) && checkCloseEnough(y, bbox_list[id].y)) {
                dragging = true;
                dragTL = true;
                console.log("Dragging top left")
            }
            // 2. top right
            else if (checkCloseEnough(x, bbox_list[id].x + bbox_list[id].w) && checkCloseEnough(y, bbox_list[id].y)) {
                dragging = true;
                dragTR = true;
                console.log("Dragging top right")
            }
            // 3. bottom left
            else if (checkCloseEnough(x, bbox_list[id].x) && checkCloseEnough(y, bbox_list[id].y + bbox_list[id].h)) {
                dragging = true;
                dragBL = true;
                console.log("Dragging bottom left")
            }
            // 4. bottom right
            else if (checkCloseEnough(x, bbox_list[id].x + bbox_list[id].w) && checkCloseEnough(y, bbox_list[id].y + bbox_list[id].h)) {
                dragging = true;
                dragBR = true;
                console.log("Dragging bottom right")
            }
            // (5.) none of them
            else {
                dragging = false;
                console.log("None")
                // handle not resizing
            }

            // ctx.clearRect(bbox_list[id].x, bbox_list[id].y, bbox_list[id].w, bbox_list[id].h);

            for(_i = 0; _b = bbox_list[_i]; _i ++) {
                if(hover && id === _i) {
                    setCurrElement(_i)
                    setShow(true);
                    ctx.fillStyle = "white";
                }
                else {
                    ctx.fillStyle = _b.color;
                }
                // ctx.fillStyle = (hover && id === _i) ? "red" : _b.color;
                ctx.fillRect(_b.x, _b.y, _b.w, _b.h);
                writeText(ctx, { text: _b.majorgroup+": "+_b.score, x: _b.x, y: _b.y });
                // setOutputGroup(_b.majorgroup);
            }
            // renderMap2(x, y);
        }
        
        canvasRef.current.onmouseup = function(e) {
            dragTL = dragBR = dragTR = dragBL = false;
            dragging = false;
            // ctx.clearRect(bbox_list[id].x, bbox_list[id].y, bbox_list[id].w, bbox_list[id].h);
            // if(dragging) {
            //     drawAnchors();
            // }
            for(_i = 0; _b = bbox_list[_i]; _i ++) {
                if(hover && id === _i) {
                    ctx.fillStyle = "white";
                }
                else {
                    ctx.fillStyle = _b.color;
                }
                // ctx.fillStyle = (hover && id === _i) ? "red" : _b.color;
                ctx.clearRect(_b.x, _b.y, _b.w, _b.h);
                ctx.fillRect(_b.x, _b.y, _b.w, _b.h);
                writeText(ctx, { text: _b.majorgroup+": "+_b.score, x: _b.x, y: _b.y });
                // setOutputGroup(_b.majorgroup);
            }
        }

        canvasRef.current.onmousemove = function(e) {
            var r = canvasRef.current.getBoundingClientRect(),
                x = e.clientX - r.left, y = e.clientY - r.top;
            if (dragTL) {
                setShow(true);
                bbox_list[id].w += bbox_list[id].x - x;
                bbox_list[id].h += bbox_list[id].y - y;
                bbox_list[id].x = x;
                bbox_list[id].y = y;
            } else if (dragTR) {
                setShow(true);
                bbox_list[id].w = Math.abs(bbox_list[id].x - x);
                bbox_list[id].h += bbox_list[id].y - y;
                bbox_list[id].y = y;
            } else if (dragBL) {
                setShow(true);
                bbox_list[id].w += bbox_list[id].x - x;
                bbox_list[id].h = Math.abs(bbox_list[id].y - y);
                bbox_list[id].x = x;
            } else if (dragBR) {
                setShow(true);
                bbox_list[id].w = Math.abs(bbox_list[id].x - x);
                bbox_list[id].h = Math.abs(bbox_list[id].y - y);
            }
            // ctx.clearRect(bbox_list[id].x, bbox_list[id].y, bbox_list[id].w, bbox_list[id].h);
            if(dragging) {
                setShow(true);
                // inEdit=true;
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                drawAnchors();
            }
            else {
                if(id === undefined) {
                    id = 1
                }
                ctx.clearRect(bbox_list[id].x, bbox_list[id].y, bbox_list[id].w, bbox_list[id].h);
                // for(_i = 0; _b = bbox_list[_i]; _i ++) {
                //     if(hover && id === _i) {
                //         ctx.fillStyle = "white";
                //     }
                //     else {
                //         ctx.fillStyle = _b.color;
                //     }
                //     // ctx.fillStyle = (hover && id === _i) ? "red" : _b.color;
                //     ctx.fillRect(_b.x, _b.y, _b.w, _b.h);
                //     // setOutputGroup(_b.majorgroup);
                // }
            }
            draw();
            console.log("empty inEdit: ", inEdit)
        }

        function draw() {
            if(hover == true) {
                ctx.fillStyle = "white";
            } else {
                ctx.fillStyle = bbox_list[id].color
            }
            ctx.globalAlpha = 0.25;
            ctx.fillRect(bbox_list[id].x, bbox_list[id].y, bbox_list[id].w, bbox_list[id].h)
            // drawAnchors();
        }

        function singleAnchor(x, y, radius) {
            ctx.fillStyle = "#FFFFFF";
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
        }

        function clearAnchor(x, y, radius) {
            ctx.globalCompositionOperation = 'destination-out'
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
        }
    
        function drawAnchors() {
            singleAnchor(bbox_list[id].x, bbox_list[id].y, closeEnough) // top left
            singleAnchor(bbox_list[id].x + bbox_list[id].w, bbox_list[id].y, closeEnough) // top right
            singleAnchor(bbox_list[id].x, bbox_list[id].y + bbox_list[id].h, closeEnough) // bottom left
            singleAnchor(bbox_list[id].x + bbox_list[id].w, bbox_list[id].y + bbox_list[id].h, closeEnough) // bottom right

            // clearAnchor(bbox_list[id].x, bbox_list[id].y, closeEnough) // top left
            // clearAnchor(bbox_list[id].x + bbox_list[id].w, bbox_list[id].y, closeEnough) // top right
            // clearAnchor(bbox_list[id].x, bbox_list[id].y + bbox_list[id].h, closeEnough) // bottom left
            // clearAnchor(bbox_list[id].x + bbox_list[id].w, bbox_list[id].y + bbox_list[id].h, closeEnough) // bottom right
        }


    }, [fileChange]);

    /*
        Draws anchor points
    */
    const drawCircle = (ctx, x, y, radius) => {
        ctx.fillStyle = "#FFFFFF";
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fill();
    }

    /*
        This function occurs after the 'Edit' buton is clicked
    */
    const drawCircles = (id, ctx) => {
        console.log("CURRENT ELEMENT STUFF: ", bboxs[id].x, bboxs[id].y, bboxs[id].w, bboxs[id].h)
        drawCircle(ctx, bboxs[id].x, bboxs[id].y, closeEnough) // top left
        drawCircle(ctx, bboxs[id].x + bboxs[id].w, bboxs[id].y, closeEnough) // top right
        drawCircle(ctx, bboxs[id].x, bboxs[id].y + bboxs[id].h, closeEnough) // bottom left
        drawCircle(ctx, bboxs[id].x + bboxs[id].w, bboxs[id].y + bboxs[id].h, closeEnough) // bottom right
        setInEdit(true);
        setShowDropDown(true);
        setDeleteButton(true);
        // inEdit = true
        console.log("INEDIT: ", inEdit)
    }

    /*
        This function saves
    */
    const dummySave = (id) => {
        setShow(false);
        setShowDropDown(false);
        setDeleteButton(false);
        // Edit json
        if(inDelete) {
            editJson[currImage].predictions.pred_boxes.splice(id, 1)
            editJson[currImage].predictions.pred_labels.splice(id, 1)
            editJson[currImage].predictions.pred_scores.splice(id, 1)
        }
        else {
            console.log("SAVED ELEMENT STUFF: ", bboxs[id].x, bboxs[id].y, bboxs[id].w, bboxs[id].h);
            var updated_box = [bboxs[id].x*6.545, bboxs[id].y*6.545, (bboxs[id].w*6.545)+(6.545*bboxs[id].x), (bboxs[id].h*6.545)+(6.545*bboxs[id].y)]
            console.log("UPDATED BOX: ", updated_box)
            editJson[currImage].predictions.pred_boxes[id] = updated_box
            editJson[currImage].predictions.pred_labels[id] = bboxs[id].majorgroup
            editJson[currImage].predictions.pred_scores[id] = bboxs[id].score
        }
        
        console.log(editJson)
        console.log("ISDELETED?: ", inDelete)
        setInDelete(false);


        // inEdit = false
        setInEdit(false);
                
        // Save json to storage
        sessionStorage.setItem("curr_json", JSON.stringify(editJson));

    }

    /*
        This function deletes
    */
    const dummyDelete = (id) => {
        // if(id == 0) {
        //     console.log(bboxs.slice(id+1));
        // } else if (id == bboxs.length-1) {
        //     console.log(bboxs.slice(0, id))
        // } else {
        //     var first_part = bboxs.slice(0, id);
        //     var second_part = bboxs.slice(id+1);
        //     var combined = first_part.concat(second_part);
        //     setBboxs(combined);
        //     console.log(bboxs)
        // }
        setInDelete(true);
        bboxs.splice(id, 1)
        setInEdit(false);
        setDeleteButton(false);
        // console.log(bboxs.slice(0, 2))
        // setShow(false);
        // setShowDropDown(false);
        // console.log("SAVED ELEMENT STUFF: ", bboxs[id].x, bboxs[id].y, bboxs[id].w, bboxs[id].h)
        // // inEdit = false
        // setInEdit(false);
    }
    /*
        This function creates a new bounding box
    */
    const dummyNew = (currElement, ctx) => {
        // Create new data point and add to bboxs list
        var to_add_x = canvasRef.current.width/2
        var to_add_y = canvasRef.current.height/2
        var to_add_w = 30
        var to_add_h = 30
        var to_add_color = "#8C8B8B" // this is the 'other' color
        var to_add_majorgroup = "Other"
        bboxs.push({x: to_add_x, y: to_add_y, w: to_add_w, h: to_add_h, color: to_add_color, majorgroup: to_add_majorgroup, score: 1})
        var updated_box = [to_add_x*6.545, to_add_y*6.545, (to_add_w*6.545)+to_add_x, (to_add_h*6.545)+to_add_y]
        editJson[currImage].predictions.pred_boxes.push(updated_box)
        editJson[currImage].predictions.pred_labels.push(to_add_majorgroup)
        console.log(editJson)
        console.log("New bboxs: ", bboxs)

        // Draw the new box
        ctx.strokeStyle = to_add_color;
        ctx.fillStyle = to_add_color;
        ctx.globalAlpha = 0.25;
        ctx.lineWidth = 2;
        // strokeRect(x, y, width, height)
        // 6.545 is our scaling from the original image dimensions (5400px x 3600px): we scale it down to (825px x 550px)
        ctx.strokeRect(to_add_x, to_add_y, to_add_w, to_add_h);
        ctx.fillRect(to_add_x, to_add_y, to_add_w, to_add_h);


        // // Set current id to new box to get ready to edit it
        // drawCircles((bboxs.length-1), ctx)

        // Save json to storage
        sessionStorage.setItem("curr_json", JSON.stringify(editJson));
    }

    /*
        This function saves a JSON output file
    */
    const dummySaveFile = () => {
        window.electronAPI.ipcR.sendProjectData(JSON.stringify(editJson, null, 4))
        const currSaveFilepath = window.electronAPI.ipcR.saveFile()
        currSaveFilepath.then(result => {
            console.log("filepath: ", result.filePath)
        })
        // var fs = require('fs');
        // fs.writeFile("test.txt", jsonData, function(err) {
        //     if (err) {
        //         console.log(err);
        //     }
        // });
        // console.log("Save path: ", currSaveFilepath)
    }
    const dummySendData = () => {
        window.electronAPI.ipcR.sendProjectData(JSON.stringify(editJson))
        // var fs = require('fs');
        // fs.writeFile("test.txt", jsonData, function(err) {
        //     if (err) {
        //         console.log(err);
        //     }
        // });
        // console.log("Save path: ", currSaveFilepath)
    }

    /*
        This function...
    */
    const handleChange = (event) => {
        setCurrMajorGroup(event);
        setConfidence(1);
        console.log(bboxs);
        bboxs[currElement].majorgroup = event;
        bboxs[currElement].color = major_group_color.get(event);
        bboxs[currElement].score = 1;
        console.log(bboxs);
    };

    const handleImageSelect = (e) => {
        const user_selected_image = img_dir_name_map.find(item => item.id == e)
        console.log("selected image: ", user_selected_image)
        // now change path
        currImageId = user_selected_image.id
        sessionStorage.setItem("curr_image_id", currImageId);
        console.log("CURR IMAGEID: ", currImageId)
        if(fileChange == false) {
            setFileChange(true);
        }
        else {
            setFileChange(false);
        }
    }


    const nextImage = () => {
        if(correctFilepaths.length-1 == currImageId) {
            console.log("No next image")
            window.electronAPI.ipcR.nextImagePopup()
        }
        else {
            console.log("Should go to next image")
            currImageId = parseInt(sessionStorage.getItem("curr_image_id"));
            currImageId += 1
            sessionStorage.setItem("curr_image_id", currImageId);
            console.log("CURR IMAGEID: ", currImageId)
            if(fileChange == false) {
                setFileChange(true);
            }
            else {
                setFileChange(false);
            }
        }
        console.log(correctFilepaths)
    }

    const prevImage = () => {
        if(currImageId == 0) {
            console.log("No prev image")
            window.electronAPI.ipcR.prevImagePopup()
        }
        else {
            console.log("Should go to prev image")
            currImageId = parseInt(sessionStorage.getItem("curr_image_id"));
            currImageId -= 1
            sessionStorage.setItem("curr_image_id", currImageId);
            console.log("CURR IMAGEID: ", currImageId)
            if(fileChange == false) {
                setFileChange(true);
            }
            else {
                setFileChange(false);
            }
        }
        console.log(correctFilepaths)
    }

    //console.log(updated) //now I can access the subgroup classification

    // subgroups.push(updated); //could so something like this to add to excel output

    // console.log(subgroups)    
    
    return (
        <section className='section'>
            <h2>Bounding Box Editor</h2>
            <Link to='/' className='btn'>
                <button>Home</button>
            </Link><br />
            <div id="wrapper">
                <canvas
                    id="bbox_canvas"
                    ref={canvasRef}
                    style={{
                        width: "825px",
                        height: "550px",
                        // background: "url('file:///C:/Users/ellyc/OneDrive/Desktop/DATA451/electron-demos/my-app-demo/public/photos/M12_2_Apr19_3.jpg')",
                        // background: "url('file:///C:/Users/edwar/Desktop/Cal\\ Poly/Ecology\\ Project/forge-test-2/src/photos_src/M12_2_Apr19_3.jpg')",
                        backgroundImage: "url(" + correctFilepaths[currImageId] + ")", //this is how you change the image!!
                        backgroundSize: "825px 550px"
                    }}
                    // align="center"
                />
                <canvas
                    id="text_canvas"
                    ref={textCanvasRef}
                    style={{
                        width: "825px",
                        height: "550px",
                        backgroundSize: "825px 550px"
                    }}
                />
                {/* <PureCanvas ref={canvasRef}
                    style={{
                        width: "800px",
                        height: "600px",
                        background: "url('./photos/M12_2_Apr19_3.jpg')",
                    }} /> */}
            </div>
            <div id="temp_legend">
                <img src={require('./photos_src/updated_label.png')} width = "150" height = "250"></img>
            </div>
            <div id="rest">
                <DropdownButton id="image_select" title="Select an Image" onSelect={handleImageSelect}>
                    {
                        img_dir_name_map.map(function(data, key) {
                            // console.log("DATA: ", data)
                            // console.log("KEY: ", key)
                            return (
                                <Dropdown.Item eventKey={data.id}>{data.file}</Dropdown.Item>
                            )
                        })
                    }
                </DropdownButton>
                {/* <select 
                    name="Images"
                    onChange={e => handleImageSelect(e)}
                    value={selectedImage}
                >
                    <option value="">Select an Image</option>
                    {
                        img_dir_name_map.map(function(data, key) {
                            console.log("DATA: ", data)
                            console.log("KEY: ", key)
                            return (
                                <option key={data.file} value={(data.path, data.id)}>{data.file}</option>
                            )
                        })
                    }
                </select> */}
                <h4>Current Image: {currImage} </h4>
                {/* <h2>?: {outputGroup}</h2> */}
                <h2>Major Group: {currMajorGroup}</h2>
                <h2>Confidence: {confidence}</h2>
                {/* <h2>Test: {bboxs.length}</h2> */}
                {
                    show && 
                    <button onClick={() => drawCircles(currElement, currCtx)}
                        className="b1"
                    >
                    Edit 
                    </button>
                }                
                {
                    show && 
                    <button onClick={() => dummySave(currElement)}
                        className="b2"
                    >
                    Save
                    </button>
                }
                {
                    deleteButton && 
                    <button onClick={() => dummyDelete(currElement)}
                        className="b2"
                    >
                    Delete
                    </button>
                }
                <br />
                <button onClick={() => dummyNew(currElement, currCtx)} 
                    className="b3"
                >
                New
                </button>
                <br /> 
                {
                    /*
                        onclick:
                            change major group text
                            change bboxs major group
                            change color
                            TODO: update count
                    */
                    showDropDown && 
                    <DropdownButton id="dropdown-button-demo" title="Change Major Group" onSelect={handleChange}>
                        <Dropdown.Item eventKey="Amphipoda">Amphipoda</Dropdown.Item>
                        <Dropdown.Item eventKey="Bivalvia">Bivalvia</Dropdown.Item>
                        <Dropdown.Item eventKey="Cumacea">Cumacea</Dropdown.Item>
                        <Dropdown.Item eventKey="Gastropoda">Gastropoda</Dropdown.Item>
                        <Dropdown.Item eventKey="Insecta">Insecta</Dropdown.Item>
                        <Dropdown.Item eventKey="Ostracoda">Ostracoda</Dropdown.Item>
                        <Dropdown.Item eventKey="Polychaeta">Polychaeta</Dropdown.Item>
                        <Dropdown.Item eventKey="Other">Other</Dropdown.Item>
                        <Dropdown.Item eventKey="Other Annelida">Other Annelida</Dropdown.Item>
                        <Dropdown.Item eventKey="Other Crustacea">Other Crustacea</Dropdown.Item>
                        <Dropdown.Item eventKey="Unknown">Unknown</Dropdown.Item>
                        <Dropdown.Item eventKey="Custom Label">Custom Label</Dropdown.Item>
                    </DropdownButton>

                }

                {/* {
                    deleteButton &&
                    <div>

                    <h2>Add Additional Label: {message}</h2>

                    <input
                    type = "text"
                    id = "message"
                    name = "message"
                    onChange = {handleChange2}
                    value = {message}
                    />

                    <button onClick={handleClick2}>Update</button>

                    </div>
                } */}


                <br />
                <button onClick={() => prevImage()}
                    className="prev-image-button"
                >
                    Prev Image
                </button>
                <button onClick={() => nextImage()}
                    className="next-image-button"
                >
                    Next Image
                </button>
                <br />
                <button onClick={() => dummySaveFile()}
                    className="save-file-button"
                >
                    Save Project
                </button>
                <button variant='contained'
                    onClick={(e) => exportToExcel(fileName, editJson)} color='primary'
                    style={{ cursor: "pointer", fontSize: 14 }}
                >
                    Download Data to Excel
                </button>
                <br />

                {/* <button onClick={() => dummySendData()}
                    className="send-data-button"
                >
                    SEND DATA
                </button> */}
            </div>
            {/* <div>
                The mouse is at position{' '}
                <b>
                    ({mousePos.x}, {mousePos.y})
                </b>
            </div> */}
        </section>
    );
};

export default ModelOutput