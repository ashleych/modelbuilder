$(document).ready(function () {
  // const ctx = document.getElementById("myChart").getContext("2d");

  var table = document.querySelector("table");
  var tableArr = []; // www.  j  a  va2s . c o m
  var tableLab = [];

  for (var i = 1; i < table.rows.length; i++) {
    tableArr.push([table.rows[i].cells[2].innerHTML]);
    tableLab.push(table.rows[i].cells[0].innerHTML);

    // var canvas = document.createElement("canvas");
    var canvas = document.createElement("div");
    canvas.setAttribute("id", "myChart" + i);
    table.rows[i].cells[4].appendChild(canvas);
  }

  tableArr.forEach(function (e, i) {
    var colors = ["blue", "green", "yellow", "orange", "red"];
    var names = ["First", "Second", "Third", "Fourth", "Fifth"];
    var chartID = "#myChart" + (i + 1);
    console.log(e);
    var roc_data_for_one_row= JSON.parse(e[0]);
   var data=[];
    roc_data_for_one_row.forEach(function(roc_,i){
      data.push({y:roc_ * 100 , name:"",color:colors[4]})
    });
    // var data=[
    //         { y: 20, name: "", color: colors[4] },
    //         { y: 7, name: "", color: colors[4] },
    //         { y: 9, name: "", color: colors[4] },
    //         { y: 1, name: "", color: colors[4] },
    //         { y: 1, name: "", color: colors[4] },
    //       ]
    $(chartID).highcharts({
      chart: {
        type: "column",
        margin: [2, 0, 2, 0],
        backgroundColor: null,
        borderWidth: 0,
        width: 120,
        height: 80,
        style: {
          overflow: "visible",
        },
      },

      title: {
        text: "",
      },
      xAxis: {
        labels: {
          enabled: false,
        },
        title: {
          text: null,
        },
        startOnTick: false,
        endOnTick: false,
        tickPositions: [],
      },
      yAxis: {
        endOnTick: false,
        startOnTick: false,
        labels: {
          enabled: false,
        },
        title: {
          text: null,
        },
        tickPositions: [0],
      },
      legend: {
        enabled: false,
      },
      credits: {
        enabled: false,
      },
      plotOptions: {
        series: {
          events: {
            legendItemClick: function (x) {
              var i = this.index - 1;
              var series = this.chart.series[0];
              var point = series.points[i];

              if (point.oldY == undefined) point.oldY = point.y;

              point.update({ y: point.y != null ? null : point.oldY });
            },
          },
        },
      },
            tooltip: {
                headerFormat: '<span style="font-size: 10px">' + 'AUC ROC:</span><br/>',
                pointFormat: '<b>{point.y}</b> %'
            },
      series: [
        {
          pointWidth: 20,
          color: colors[0],
          showInLegend: false,
          data: data,
        },
        { color: "blue" },
        { color: "green" },
        { color: "yellow" },
        { color: "orange" },
        { color: "red" },
      ],
    });
  });
});
