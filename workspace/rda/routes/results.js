var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/main', function(req, res, next) {
  res.render('result', { type: 'main'});
});
router.get('/classification', function(req, res, next) {
  res.render('result', { type: 'classification' , data: { isEng: false}});
});
router.get('/feature', function(req, res, next) {
    res.render('result', { type: 'feature'});
  });
router.get('/cluster', function(req, res, next) {
    res.render('result', { type: 'cluster'});
  });
router.get('/corr', function(req, res, next) {
    res.render('result', { type: 'corr'});
  });  
module.exports = router;
