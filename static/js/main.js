$(document).ready(function() {
  $("#simulated_annealing_params").fadeOut();
  $("#genetic_algorithm_params").fadeOut();
  $("#nb_machines").change(function() {
    if ($("#algorithm").val() === "johnson") {
      if ($("#nb_machines") !== 2) {
        $("#nb_machines").val(2);
      }
    } else {
      if ($(this).val() <= 2) {
        $("#nb_machines").val(2);
      }
    }
  });

  $("#algorithm").change(function() {
    if ($(this).val() === "johnson") {
      $("#nb_machines").val(2);
    }
    if ($(this).val() === "simulated-annealing") {
      $("#simulated_annealing_params").fadeIn();
    } else {
      $("#simulated_annealing_params").fadeOut();
    }

    if ($(this).val() === "genetic-algorithm") {
      $("#genetic_algorithm_params").fadeIn();
    } else {
      $("#genetic_algorithm_params").fadeOut();
    }


  });

  $("#solve").on("click", function() {
    
    if ($("#algorithm").val() === "simulated-annealing") {
      var data_tbs = JSON.stringify({
        algorithm: $("#algorithm").val(),
        data: $("#data").val(),
        nb_machines: $("#nb_machines").val(),
        nb_jobs: $("#nb_jobs").val(),
        ti: $("#init_temp").val(),
        tf: $("#f_temp").val(),
        alpha: $("#alpha").val()
      });
    } else if ($("#algorithm").val() === "genetic-algorithm") {
      var data_tbs = JSON.stringify({
        nograph: true,
        algorithm: $("#algorithm").val(),
        data: $("#data").val(),
        nb_machines: $("#nb_machines").val(),
        nb_jobs: $("#nb_jobs").val(),
        population_number: $("#pop_number").val(),
        it_number: $("#it_number").val(),
        p_crossover: $("#p_crossover").val(),
        p_mutation: $("#p_mutation").val()
      });
    } else {
      var data_tbs = JSON.stringify({
        algorithm: $("#algorithm").val(),
        data: $("#data").val(),
        nb_machines: $("#nb_machines").val(),
        nb_jobs: $("#nb_jobs").val()
      });
    }
    
    $.ajax({
      url: "/solve",
      dataType: "json",
      type: "post",
      contentType: "application/json",
      data: data_tbs,
      processData: false,
      success: function(data, textStatus, jQxhr) {
        if (data["graph"] === null) {
          $("#sequence").text(data["opt_seq"]);
          $("#opt_makespan").text(data["optim_makespan"]);
          var time_str = data["t_time"].toString() + " " + data["tt"].toString();
          $("#time").text(time_str);
        } else {
          Plotly.newPlot("gantt", JSON.parse(data["graph"]), {});
          $("#sequence").text(data["opt_seq"]);
          $("#opt_makespan").text(data["optim_makespan"]);
          var time_str = data["t_time"].toString() + " " + data["tt"].toString();
          $("#time").text(time_str);
        }
        
      },
      error: function(jQxhr, textStatus, errorThrow) {
        console.log(textStatus);
      }
    });
  });
  $("#gen_random").on("click", function() {
    $.ajax({
      url: "/random",
      dataType: "text",
      type: "post",
      contentType: "application/json",
      data: JSON.stringify({
        nb_machines: $("#nb_machines").val(),
        nb_jobs: $("#nb_jobs").val()
      }),
      processData: false,
      success: function(data, textStatus, jQxhr) {
        $("#data").text(data);
      },
      error: function(jQxhr, textStatus, errorThrow) {
        alert("AJAX ERROR");
      }
    });
  });

  $("#gantt_toggle").click(function() {
    $("#gantt").fadeToggle();
  });
});
