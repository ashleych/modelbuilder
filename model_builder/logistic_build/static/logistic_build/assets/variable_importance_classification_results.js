$(document).ready(function () {
  // const ctx = document.getElementById("myChart").getContext("2d");

  var table = document.querySelector("#model_spec_table");
  var tableArr = []; // www.  j  a  va2s . c o m
  var features = [];
  var data = [];
  var colors = ["blue", "green", "yellow", "orange", "red"];
  for (var i = 1; i < table.rows.length; i++) {
    // tableArr.push([table.rows[i].cells[1].innerHTML]);
    // features.push(table.rows[i].cells[0].innerHTML);
    feature = table.rows[i].cells[0].textContent.trim();
    abs_coefficient = Math.abs(table.rows[i].cells[1].textContent.trim());
    data.push({ y: abs_coefficient, name: feature, color: '#16838E' });
  }
  var dataSortByAbsCoefficient = data.sort(function (a, b) {
    return b.y - a.y;
  });

  var data_sliced = dataSortByAbsCoefficient.slice(0, 10);

  var chartID = "#variable_importance_chart";
      $(chartID).highcharts({
        chart: {
          type: "bar",
          // margin: [2, 0, 2, 0],
          backgroundColor: null,
          // borderWidth: 0,
          width: 500,
          // // height: 80,
          style: {
            overflow: "visible",
          },
        },

        title: {
          text: "",
        },
        xAxis: {
 
          labels: {
            enabled: true,
             formatter: function() { return data_sliced[this.value].name;}
          },
          title: {
            text: null,
          },
          startOnTick: false,
          endOnTick: false,
          // tickPositions: [],
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
                  headerFormat: '<span style="font-size: 10px">' + 'Top feature:</span><br/>',
                  pointFormat: '<b>{point.name}</b> '
              },
        series: [
          {
            pointWidth: 20,
            color: '#16838E',
            showInLegend: true,
            data: data_sliced,
          },
          // { color: "blue" },
          // { color: "green" },
          // { color: "yellow" },
          // { color: "orange" },
          // { color: "#16838E" },
        ],
      });
});
