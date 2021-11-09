const express = require('express');
const router = express.Router();

var db_config = require('../config/database.js');
var conn = db_config.init();
db_config.connect(conn);
//pepper테이블 col 추출
var col=[];
conn.query('SHOW COLUMNS FROM pepper_data', function(err, rows, fields){ 
    for( data of rows){
        col=[...col, data.Field]
    }
});

/*고추유전자원 데이터 조회 */
router.get('/list/:page', function (req, res) {
    var page = req.params.page;
    var sql = 'SELECT * FROM pepper_data';    
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data.ejs', { 
                path:"/list",
                type:"pepper", 
                method:"get",
                data : result, 
                col: col, 
                page:page, 
                length: result.length-1
            });
        }                
    });
});

/*고추유전자원 데이터 검색 초기화면 */
router.get('/search/:page', function (req, res) {
    var page = req.params.page;
    var sql = 'SELECT * FROM pepper_data';    
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data-search.ejs', { 
                path:"/search", 
                type:"pepper",
                method:"get",
                data : result, 
                col: col, 
                page:page, 
                length: result.length-1
            });
        }                
    });
});

/*고추유전자원 데이터 검색 결과화면 */
router.post('/searchResult/:page', function (req, res) {
    var page = req.params.page;
    var searchCol=req.body;
    var sql = `SELECT * FROM pepper_data WHERE ${searchCol.col} = "${searchCol.search}"`;    

    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data-search.ejs', { 
                path:"/searchResult", 
                type:"pepper",
                method:"post", 
                data : result, 
                col: col, 
                page:page, 
                length: result.length-1, 
                search:{col:searchCol.col, search:searchCol.search}});
        }                
    });

});

/*고추유전자원 데이터 다중검색 초기화면 */
router.get('/advSearch/:page', function (req, res) {
    var page = req.params.page;
    var sql = 'SELECT * FROM pepper_data';    
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data-multipleSearch.ejs', { 
                path:"/advSearch", 
                type:"pepper",
                method:"get",
                data : result, 
                col: col, 
                page:page, 
                length: result.length-1
            });
        }                

    });
});

/*고추유전자원 데이터 다중검색 결과화면 */
router.post('/advSearchResult/:page', function (req, res) {
    var page = req.params.page;
    var searchCol=req.body;
    var sql = `SELECT * FROM pepper_data WHERE `;
    if(searchCol.col!="select column"){
        sql=sql+`${searchCol.col} = "${searchCol.search}"`;} 
    if(searchCol.col2!="select column"){
        sql=sql+` AND ${searchCol.col2} = "${searchCol.search2}"`;
    } 
    if(searchCol.col3!="select column"){
        sql=sql+` AND ${searchCol.col3} = "${searchCol.search3}"`;
    }
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data-multipleSearch.ejs', { 
                path:"/advancedSearch",
                type:"pepper",
                method:"post", 
                data : result,
                col: col, page:page, 
                length: result.length-1, 
                search:{
                    col:searchCol.col, 
                    search:searchCol.search,
                    col2:searchCol.col2, 
                    search2:searchCol.search2,
                    col3:searchCol.col3, 
                    search3:searchCol.search3,
                }});
        }                
    });

});

/*고추유전자원 데이터 column 초기화면 */
router.get('/column', function (req, res) {
    res.render('data-featureSearch.ejs', { path:"/column", type:"pepper", method:"get", col: col})
});

/*고추유전자원 데이터 column 결과화면 */
router.post('/column', function (req, res) {
    var selCol = req.body.col;
    var sql = `SELECT ${selCol} , COUNT(${selCol}) AS cnt FROM pepper_data GROUP BY ${selCol} order by cnt desc`;  
    var colGroup=[]
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            for( data of result){
                colGroup=[...colGroup, {col : data[selCol], cnt: data.cnt}]
            }
            res.render('data-featureSearch.ejs', {
                path:"/column", 
                type:"pepper", 
                method:"post", 
                colGroup : colGroup, 
                selCol: selCol, 
                col: col
            });
        }                
    });
   
});

/*고추유전자원 데이터 column 검색 결과화면 */
router.post('/columnResult/:page', function (req, res) {
    var page = req.params.page;
    var searchCol=req.body;
    var sql = `SELECT * FROM pepper_data WHERE ${searchCol.col} = "${searchCol.search}"`;   
    var sql_col = `SELECT ${searchCol.col} , COUNT(${searchCol.col}) AS cnt FROM pepper_data GROUP BY ${searchCol.col} order by cnt desc`;  
    var colGroup=[]
    conn.query(sql_col, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            for( data of result){
                colGroup=[...colGroup, {col : data[searchCol.col], cnt: data.cnt}]
            }
        }                
    });
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data-featureSearch.ejs', { 
                path:"/columnResult",
                type:"pepper",
                method:"post", 
                data : result,
                col: col, 
                colGroup : colGroup,
                page:page, 
                length: result.length-1, 
                selCol: searchCol.col,
                search:{
                    col:searchCol.col, 
                    search:searchCol.search,
                }});
        }                
    }); 

});
module.exports = router;