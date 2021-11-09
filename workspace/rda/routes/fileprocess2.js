var express = require('express');
const { spawn } = require('child_process');
var multer = require('multer');
var path = require('path');
var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'files/')
    },
    filename: function (req, file, cb) {
        cb(null, `${file.originalname}__${Date.now()}`)
    }
});
var upload = multer({ storage: storage, preservePath: path.resolve('files/') });
var fs = require('fs');
var mime = require('mime');
const { json } = require('express');
const { route } = require('.');
const getDownloadFilename = require('./lib/getDownloadFilename').getDownloadFilename;
var router = express.Router();
/* GET home page. */

/*Feat : Deep-learning run */
router.post('/', upload.single('input-file'), async (req, res) => {
    req.connection.setTimeout(60 * 30 * 1000) // set timeout 3 min
    var dataToSend;
    var absolutePath = upload.preservePath;
    var complete=0;
    //correlation python 실행
    spawn('python3', ['../pythonScripts/correlation.py', absolutePath, req.file.filename]);
    //feature_importance python 실행
    spawn('python3', ['../pythonScripts/feature_importance.py', absolutePath, req.file.filename, 3]);
    //clustering python 실행
    spawn('python3', ['../pythonScripts/clustering.py', absolutePath, req.file.filename]);
    //classification python 실행
    const python_classification = spawn('python3', ['../pythonScripts/classification.py', absolutePath, req.file.filename]);
    python_classification.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
        var result = req.file;
        Object.assign(result,
            { filepath: absolutePath },
            { resultfilename: "10cv_acc_" + req.file.filename + "_.csv" },
            { resultfilename2: "importance_score_result_" + req.file.filename + "_.csv" },
            { resultfilename3: "importance_score_" + req.file.filename + "_.csv" },
            { resultfilename4: "cluster_data_" + req.file.filename + "_.csv" },
            { resultfilename5: "clustering_score_" + req.file.filename + "_.csv" },
            { resultfilename6: "feature_selection_result_" + req.file.filename + "_.csv" },
            { resultfilename7: "feature_selection_" + req.file.filename + "_.csv" },
            { resultfilename8: "clustering_similarity_score_" + req.file.filename + "_.csv" },
            { cluster_img1: "Silhouette_score_" + req.file.filename + "_.png" },
            { cluster_img2: "Dendrogram_" + req.file.filename + "_.png" },
            { feature_img1: "rf_feature_importance" + req.file.filename + "_.png" },
            { feature_img2: "rf_feature_importance_barplot" + req.file.filename + "_.png" },
            { corr_img1: "pearson_corr_heatmap_" + req.file.filename + "_.png" },
            { corr_img2: "spearman_corr_heatmap_" + req.file.filename + "_.png" },
            { corr_img3: "pearson_corr_tri_heatmap_" + req.file.filename + "_.png" },
            { corr_img4: "spearman_corr_tri_heatmap_" + req.file.filename + "_.png" },
            { corr_img5: "pairplot_" + req.file.filename + "_.png" },
            { pca_img: "pca_" + req.file.filename + "_.png" });
        var default_class = { kernel: "linear", n_neigbors: "3", n_estimator: "100", criterion: "gini" };
        var default_feature = { feature_selection_num: "3" };
        var default_cluster = { kmeans_n_clusters: "3", max_iter: "300", eps: "0.5", min_samples: "5", hc_n_clusters: "3" };

        global.param_class = default_class;
        global.param_feature = default_feature;
        global.param_cluster = default_cluster;
        global.maindata = result;
        res.render('result', { maindata: maindata, type:"main" });

    });

});
/*Feat : file-download */
router.get('/:fileName', function (req, res) {
    const fileName = req.params.fileName;
    var filePath = path.resolve('public/files/') + "/" + decodeURI(fileName);
    try {
        if (fs.existsSync(filePath)) {
            var filename = path.basename(filePath);
            var mimetype = mime.getType(filePath);

            res.setHeader('Content-disposition', 'attachment; filename=' + getDownloadFilename(req, filename));
            res.setHeader('Content-type', mimetype);

            var filestream = fs.createReadStream(filePath);
            filestream.pipe(res);

        } else {
            res.send('해당 파일이 없습니다.');
            return;
        }
    } catch (e) {
        console.log(e);
        res.send('파일을 다운로드하는 중에 에러가 발생하였습니다.');
        return;
    }
});
/*Feat : Deep-learning run with custom params*/
router.post('/classification-run', async (req, res) => {
    const python_reclassification = spawn('python3', ['../pythonScripts/classification_terminal_new.py', maindata.filepath, maindata.filename, req.body.kernel, req.body.n_neigbors, req.body.n_estimator, req.body.criterion]);
    param_class = req.body;
    python_reclassification.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
        res.render('result', { maindata: maindata, type:"classification" });
    });
});
router.post('/feature-run', async (req, res) => {
    param_feature = req.body;
    const python3 = spawn('python3', ['../pythonScripts/feature_selection_terminal.py', maindata.filepath, maindata.filename, req.body.feature_selection_num]);
    python3.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
        res.render('result', { maindata: maindata, type:"feature" });
    });
});
router.post('/clustering-run', async (req, res) => {
    param_cluster = req.body;
    const python_recluster = spawn('python3', ['../pythonScripts/clustering_terminal_new.py', maindata.filepath, maindata.filename, req.body.kmeans_n_clusters, req.body.max_iter, req.body.eps, req.body.min_samples, req.body.hc_n_clusters]);
    python_recluster.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
        const python_repca = spawn('python3', ['../pythonScripts/pca_terminal.py', maindata.filepath, maindata.filename]);
        python_repca.on('close', (code) => {
            console.log(`child process close all stdio with code ${code}`);
            res.render('result', { maindata: maindata, type:"cluster" });
        });
    });
});
module.exports = router;
