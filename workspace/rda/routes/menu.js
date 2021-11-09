var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/about', function(req, res, next) {
  res.render('about', { title: 'Express' });
});

router.get('/help', function(req, res, next) {
    res.render('help', { title: 'Express' });
});

router.get('/data', function(req, res, next) {
    res.render('data', { title: 'Express' });
});

module.exports = router;
